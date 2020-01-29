import sys
import math
import numpy as np
import cv2
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

DEG2RAD = math.pi / 180.0
def ObjectiveFunction(free_rate, velocity_rate, heading_rate, only_free_space):
    # 1) free space rate, 2) velocity 3) heading to goal
    c_gain = 100.0
    v_gain = 0.01#0.01
    h_gain = 50.0#50
    
    weighted_free_rate = np.asarray([c_gain*free_rate[i] if i in only_free_space else free_rate[i] for i in range(free_rate.shape[0])])
    total_fitness = weighted_free_rate+v_gain*velocity_rate+h_gain*heading_rate
    return total_fitness

def DynamicWindow_withoutTarget(discrete_steering, curr_velo, curr_steer, peaks, peak_headings, navi_info, arrow, img, img_a):
    

    if navi_info == 'R':
        target_heading = peak_headings[-1]
        target_arrow = arrow[-1]
    elif navi_info == 'L':
        target_heading = peak_headings[0]
        target_arrow = arrow[0]
    else:
        middle_idx = len(peaks)/2 if len(peaks)%2==0 else (len(peaks)-1)/2
        target_heading = peak_headings[int(middle_idx)]
        target_arrow = arrow[int(middle_idx)]
    
    
    #1: Set a range of the search space. a.k.a Dynamic Window
    search_space_range_s = 20
    steer_resol = 30
    search_space_s = np.clip([curr_steer + (i-search_space_range_s/2)*steer_resol for i in range(search_space_range_s+1)], -540, 540) # [-540, 540]
    
    
    window_max_steer = np.max(search_space_s)
    window_min_steer = np.min(search_space_s)

    min_velo = 3.0#1.39
    max_velo = 12.0
    search_space_range_v = 10
    velo_resol = 0.5
    search_space_v = np.clip([curr_velo + (i-search_space_range_v/2)*velo_resol for i in range(search_space_range_v+1)], min_velo, max_velo) # [m/s]
    
    window_max_velo = np.max(search_space_v)
    window_min_velo = np.min(search_space_v)

    
    search_space = [(s,v) for s in np.unique(search_space_s) for v in np.unique(search_space_v)]
    
    minmax = search_space.index((window_min_steer, window_max_velo))
    
    midmin = search_space.index((curr_steer, window_min_velo))
    midmax = search_space.index((curr_steer, window_max_velo))

    maxmax = search_space.index((window_max_steer, window_max_velo))

    #2: Find the optimal control (velocity, steering) profile over a defined objective function.
    SensingRange_x = 2
    SensingRange_y = 10 # ms: sensing center is in front of the agent.
    cX = (int)(img.shape[1]/2) + SensingRange_x
    cY = (int)(img.shape[0]-1) + SensingRange_y
    # cY = (int)(img.shape[0]/2)+SensingRange #+ 169
    
    draw_lines = []
    
    dt = .05
    steering_ratio = 14.60
    max_t = 1.0
    wheel_base = 2.845
    idx = 0
    bias_r = 16
    bias_l = 15
    

    collision_included = np.zeros(len(search_space))
    free_rates = np.zeros(len(search_space))
    velocity_cands = np.zeros(len(search_space))

    yaw_trj_errors = np.zeros((len(search_space)))

    normalized_img = (img/np.max(img))[:,:,0] #### 1: free space, 0:obstacle
    normalized_img_obs = 1-normalized_img #### 0: free space, 1:obstacle

    ''' Draw ray sensor on the image '''
    for cand in search_space:
        # cand[0]: steering, cand[1]: velocity
        yaw = 0
        real2pix = 10 # scale factor
        step = 0
        collision_rate = 0
        free_rate = 0
        obs_rate = 0
        
        trajectory = []
        # trajectory.append((cX, cY))
        now_dangerous = False

        while step*dt < max_t:
            step += 1
            current_yaw = DEG2RAD*cand[0]/steering_ratio# This is because of the vehicle,
            
            slip_beta = math.atan2(math.tan(current_yaw), 2)
            dX = cand[1]*math.cos(yaw + slip_beta)*dt
            dY = cand[1]*math.sin(yaw + slip_beta)*dt
            yaw += dt*cand[1]*math.tan(current_yaw)*math.cos(slip_beta) / wheel_base
            
            yaw_trj_errors[idx] = abs(yaw - target_heading)

            tX = (int)(cX + real2pix * step*dY)
            tY = (int)(cY - real2pix * step*dX)
        
            if tX >= (int)(img.shape[1]) or tX < 0 or \
                tY >= (int)(img.shape[0]) or tY < 0:
                # print("oh")
                continue
            
            if list(img[tY][tX]) == [0, 0, 0]: # encounter the occupied pixel(black)
                now_dangerous = True
               
            if now_dangerous:                
                collision_rate +=1

            trajectory.append((tX, tY))
        
        for point in trajectory:
            free_rate += np.sum(normalized_img[point[1],point[0]-bias_l:point[0]+bias_r+1])
            obs_rate += np.sum(normalized_img_obs[point[1],point[0]-bias_l:point[0]+bias_r+1])
        
        draw_lines.append(trajectory)
        
        collision_included[idx] = collision_rate
        free_rates[idx] = free_rate/(free_rate+obs_rate)
        
        velocity_cands[idx] = cand[1]
        idx += 1
        
    
    free_rates = free_rates/np.max(free_rates)
    only_free_space = np.where(free_rates>=0.99)
    only_free_space = np.asarray(only_free_space)[-1]
    velocity_cands = velocity_cands/np.max(velocity_cands)
    normalized_trj_yaw_error = (-1)*(yaw_trj_errors/np.max(yaw_trj_errors))
    
    total_fitness = ObjectiveFunction(free_rates, velocity_cands, normalized_trj_yaw_error, only_free_space)
    optimal_idx = np.argmax(total_fitness)

    for i, line in enumerate(draw_lines):
        for idx in range(len(line)):
            if idx+1 == len(line):
                break
            # if idx % 1 ==0:
            cv2.line(img_a, (line[idx][0], line[idx][1]), \
                        (line[idx+1][0], line[idx+1][1]), (0, 255, 0), 1)
        
    for j in range(len(draw_lines[optimal_idx])):
        if j+1 == len(draw_lines[optimal_idx]):
            break
        cv2.line(img_a, (draw_lines[optimal_idx][j][0]+bias_r, draw_lines[optimal_idx][j][1]), \
                            (draw_lines[optimal_idx][j+1][0]+bias_r, draw_lines[optimal_idx][j+1][1]), (0, 0, 255), 2)
        cv2.line(img_a, (draw_lines[optimal_idx][j][0]-bias_l, draw_lines[optimal_idx][j][1]), \
                            (draw_lines[optimal_idx][j+1][0]-bias_l, draw_lines[optimal_idx][j+1][1]), (0, 0, 255), 2)

    
    


    mid_x = int((draw_lines[minmax][-1][0] + draw_lines[maxmax][-1][0]) /2)
    mid_y = int((draw_lines[minmax][-1][1] + draw_lines[maxmax][-1][1]) /2)
    draw_theta = math.atan2(-(draw_lines[midmax][-1][1] - mid_y), (draw_lines[midmax][-1][0] - mid_x + 1e-5))
    length_h = math.sqrt((draw_lines[midmax][-1][1] - mid_y)**2 + (draw_lines[midmax][-1][0] - mid_x)**2)
    a = (int(draw_lines[minmax][-1][0] + length_h*math.cos(draw_theta)), int(draw_lines[minmax][-1][1] - length_h*math.sin(draw_theta)))
    b = (int(draw_lines[maxmax][-1][0] + length_h*math.cos(draw_theta)), int(draw_lines[maxmax][-1][1] - length_h*math.sin(draw_theta)))


    cv2.line(img_a, (draw_lines[minmax][-1][0], draw_lines[minmax][-1][1]), \
                            (draw_lines[maxmax][-1][0], draw_lines[maxmax][-1][1]), (255, 255, 0), 2)
    cv2.line(img_a, (draw_lines[minmax][-1][0], draw_lines[minmax][-1][1]), \
                            (a[0], a[1]), (255, 255, 0), 2)                            
    cv2.line(img_a, (draw_lines[maxmax][-1][0], draw_lines[maxmax][-1][1]), \
                            (b[0], b[1]), (255, 255, 0), 2)                            
    cv2.line(img_a, (a[0], a[1]), (b[0], b[1]), (255, 255, 0), 2)                
                                        
    cv2.arrowedLine(img_a, (target_arrow[0][0], target_arrow[0][1]),\
                           (target_arrow[-1][0], target_arrow[-1][1]), (255, 0, 0), 2, tipLength = 1.0) 
    # cv2.imshow("Dynamic Window", img)
    # cv2.waitKey(1)

    return search_space[optimal_idx][0], search_space[optimal_idx][1], img_a
    # return 0.55*search_space[optimal_idx][0]/540, search_space[optimal_idx][1], img_a #ms: CARLA
