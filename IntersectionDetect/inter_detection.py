#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function


# -- find carla module ---------------------------------------------------------
# ==============================================================================
# ==============================================================================


import glob
import os
import sys

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import pygame
from pygame.locals import K_d
from pygame.locals import K_a
from pygame.locals import K_w

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
from sklearn import mixture
from scipy.signal import find_peaks

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import argparse
import collections
import datetime
import logging
import math
import random

import math

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from dynamic_window import DynamicWindow_withoutTarget


histo = None
fig = None
DISCRETE_RANGE = 61
# DISCRETE_RANGE = 31
DEG2RAD = math.pi / 180.0
MAVG_PARAM = 1
imageCall = None
curr_steer = None
curr_velo = None
imageCall_alpha = None

REAL_VEHICLE_GAIN = 0.5


NAVI_INFO = {'Left': 'L', 'Straight': 'S', 'Right': 'R', 'None': 'None'}
CONFIDENCE_TOR = 20
DWA = True

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def image_callback(msg):
    global imageCall
    imageCall = np.fromstring(msg.data, dtype = np.uint8).reshape(96, 96, 3)
    
def image_alpha_callback(msg):
    global imageCall_alpha
    imageCall_alpha = np.fromstring(msg.data, dtype = np.uint8).reshape(96, 96, 3)

def steer_callback(msg):
    global curr_steer #DEG
    curr_steer = msg.data[0]
    # print(curr_steer)
def velo_callback(msg):
    global curr_velo
    curr_velo = msg.data[0]

def inter_detect(histogram, distance = 20, height = 0.45):#distance = 20, height = 0.45
    dist_func = histogram/np.max(histogram)
    dist_func = moving_average(dist_func, MAVG_PARAM)
    
    local_optima, _ = find_peaks(dist_func, distance = distance, height = (height,))# parameter: distance, height(bound_range of peaks)
    # local_optima = 2*(local_optima-45)
    return dist_func, local_optima


def SDF(max_steering, k):#Steering_Distribution_Function, abs(x)**(1/k)
    half_range = int((DISCRETE_RANGE-1)/2)
    positive_steering = np.zeros(half_range)
    for i in range(half_range):
        positive_steering[i] = max_steering*((i+1)/half_range)**k
    negative_steering = positive_steering*(-1)
    steering_cands = np.append(np.flip(negative_steering), np.asarray([0]))
    steering_cands = np.append(steering_cands, positive_steering)
    # print(steering_cands)
    # print(len(moving_average(steering_cands, 5)))
    return steering_cands


def image_preprocessing(img):
    image_input = np.copy(img)    
    image_input[88:91,39:64,0] = 255.#VEHICLE
    image_input[88:91,39:64,1] = 100.
    image_input[88:91,39:64,2] = 180.

    image_input[91,38:65,0] = 255.#VEHICLE
    image_input[91,38:65,1] = 100.
    image_input[91,38:65,2] = 180.

    image_input[92:,36:66,0] = 255.#VEHICLE
    image_input[92:,36:66,1] = 100.
    image_input[92:,36:66,2] = 180.
    
    image_input[-1:,35:67,0] = 255.#VEHICLE
    image_input[-1:,35:67,1] = 100.
    image_input[-1:,35:67,2] = 180.
    return image_input
    # cv2.imshow('wow', image_input)
    # cv2.waitKey(1)

def future_trj_casting(image, discrete_steering):
    
    draw_image = np.copy(image)# ms: it is dangerous to use an input variable directly, cause python has very ambiguous rule about the pointer(?)

    SensingRange = 40 # ms: sensing center is in front of the agent.
    draw_image[:,:,0][draw_image[:,:,0]<=100.] = 0.
    draw_image[:,:,1][draw_image[:,:,1]<=100.] = 0.
    draw_image[:,:,2][draw_image[:,:,2]<=100.] = 0.
   

    global histo, fig
    cX = (int)(draw_image.shape[1]/2)
    cY = (int)(draw_image.shape[0]/2)+SensingRange #+ 169
    # print(cX, cY)
    steer_list = [] # -540 < steer < 540 --> 1080
    trajectory_set = []
    future_trj_dist = np.zeros(DISCRETE_RANGE)
    target_headings = np.zeros(DISCRETE_RANGE)

    idx = 0

    steering_ratio = 14.60
    current_velocity = 1.0
    dt = 1
    wheel_base = 2.845


    ''' Draw ray sensor on the image '''
    for steering in discrete_steering:
        steer_list.append(steering)
        currR = 0
        dR = 0
        check_done = False
        
        yaw = 0
        real2pix = 2 # scale factor

        trajectory = []
        trajectory.append((cX, cY))
        while True:
            dR += 1
            current_yaw = DEG2RAD*steering/steering_ratio
            slip_beta = math.atan2(math.tan(current_yaw), 2)
            dX = current_velocity*math.cos(yaw + slip_beta)*dt
            dY = current_velocity*math.sin(yaw + slip_beta)*dt
            yaw += dt*current_velocity*math.tan(current_yaw)*math.cos(slip_beta) / wheel_base


            tX = (int)(cX + real2pix *dR*dY)
            tY = (int)(cY - real2pix *dR*dX)
            
            ##this line  is for checking the lines overpass the image;
            if tX >= (int)(draw_image.shape[1]) or tX < 0 or \
                tY >= (int)(draw_image.shape[0]) or tY < 0:
                break

            # print(draw_image[tY][tX])
            if (not check_done):
                if list(draw_image[tY][tX]) != [255, 255, 255]: # encounter the occupied pixel(black(000))
                    check_done = True
                    currR = dR
                    break
                else:
                    trajectory.append((tX, tY))

        if (not check_done):
            trajectory.append((tX,tY))
            currR = dR
        
        trajectory_set.append(trajectory)
        future_trj_dist[idx] = currR
        target_headings[idx] = yaw# Save the last yaw
        
        idx += 1
    
    
    normalized_range, local_peaks = inter_detect(future_trj_dist)
    local_peaks = np.asarray(local_peaks)
    """ reasonable peaks """
    reasonable_peaks = np.asarray([i for i in local_peaks if future_trj_dist[i] > CONFIDENCE_TOR])
    isIntersection = True if reasonable_peaks.shape[0] >= 2 else False
    

    rp_target_headings = np.asarray([target_headings[i] for i in reasonable_peaks])    
    lp_target_headings = np.asarray([target_headings[i] for i in local_peaks])

    rp_arrow = np.asarray([trajectory_set[i][-5:-1] for i in reasonable_peaks])
    lp_arrow = np.asarray([trajectory_set[i][-5:-1] for i in local_peaks])

    histo.set_ydata(normalized_range)
    fig.canvas.draw()

    RAY_VIS = True
    if RAY_VIS:
        for trj in trajectory_set:
            for idx in range(len(trj)):
                if idx+1 == len(trj):
                    break
                else: 
                    cv2.line(draw_image, (trj[idx][0], trj[idx][1]), \
                                        (trj[idx+1][0], trj[idx+1][1]), (0, 255, 0), 1)

        cv2.imshow("view", draw_image)
        cv2.waitKey(1)
        print("Intersection: {}".format(isIntersection))
    return local_peaks, reasonable_peaks, rp_target_headings, lp_target_headings, isIntersection, rp_arrow, lp_arrow

def pubImage(publisher, nd_image):
    
    out_msg = Image()
    out_msg.height = nd_image.shape[0]
    out_msg.width = nd_image.shape[1]
    # print(out_vis_alpha_msg.height, out_vis_alpha_msg.width)
    out_msg.step = nd_image.strides[0]
    out_msg.encoding = 'bgr8'
    out_msg.header.frame_id = 'map'
    out_msg.header.stamp = rospy.Time.now()
    out_msg.data = nd_image.flatten().tolist()
    
    publisher.publish(out_msg)

def main():
    
    global fig, histo
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_title("Intersection")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Histogram")
    
    discrete_steering = SDF(max_steering = 200, k=2)
    histo, = ax.plot(discrete_steering, np.zeros((discrete_steering.shape[0],)), c = '#00ff00', lw=3, alpha=0.5)
    # histo, = ax.plot(np.linspace(-90,90,DISCRETE_RANGE+1-MAVG_PARAM), np.zeros((DISCRETE_RANGE+1-MAVG_PARAM,)), c = 'b', lw=3, alpha=0.5)
    # ax.set_xlim(0, DISCRETE_RANGE)
    ax.set_ylim(0,1.1)
    plt.ion()
    plt.show()

    rospy.init_node('intersection_detect_n_driving_using_DWA')
    print("Start!")
    pub_image_result = rospy.Publisher('front_usb_cam/image_result', Image, queue_size=1)
    pub_dwa_steer = rospy.Publisher('dwa_steering', Float32MultiArray, queue_size=1)
    sub_image = rospy.Subscriber('front_usb_cam/image_ss', Image, image_callback)    # new_vehicle)
    sub_image_alpha = rospy.Subscriber('front_usb_cam/image_alpha', Image, image_alpha_callback)    # new_vehicle)
    sub_steer = rospy.Subscriber('/currentSteer', Float32MultiArray, steer_callback)    # new_vehicle)
    sub_velo = rospy.Subscriber('/CanVelData', Float32MultiArray, velo_callback)    # new_vehicle)
    
    optimal_steer = None
    optimal_velo = None
    global_navi_info = None

    showing_img = None

    pygame.init()
    pygame.display.set_mode((100,100))

    global imageCall, curr_steer, curr_velo, imageCall_alpha
    
    ''' For Memorable '''
    m_l_peaks = None
    m_r_peaks = None
    m_rp_headings = None
    m_lp_headings = None
    m_rp_arrows = None
    m_lp_arrows = None

    while not rospy.is_shutdown():
        if imageCall is not None and imageCall_alpha is not None:
            l_peaks, r_peaks, rp_headings, lp_headings, isIntersection, rp_arrows, lp_arrows = future_trj_casting(imageCall, discrete_steering)

            '''Determine the heading by the given navi info'''
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                if keys[K_d]:
                    global_navi_info = NAVI_INFO['Right']
                elif keys[K_w]:
                    global_navi_info = NAVI_INFO['Straight']
                elif keys[K_a]:
                    global_navi_info = NAVI_INFO['Left']

            if l_peaks.shape[0] !=0:
                m_l_peaks = np.copy(l_peaks)
                m_lp_headings = np.copy(lp_headings)
                m_lp_arrows = np.copy(lp_arrows)

            if r_peaks.shape[0] !=0:
                m_r_peaks = np.copy(r_peaks)
                m_rp_headings = np.copy(rp_headings)
                m_rp_arrows =np.copy(rp_arrows)

            steer_command = Float32MultiArray()


            if DWA and curr_steer is not None and curr_velo is not None:
                copied_img = image_preprocessing(imageCall)
                copied_img_alpha = np.copy(imageCall_alpha)
                if m_r_peaks.shape[0] != 0:
                    optimal_steer, optimal_velo, showing_img = DynamicWindow_withoutTarget(discrete_steering, curr_velo, curr_steer, m_r_peaks, \
                                                            m_rp_headings, global_navi_info, m_rp_arrows, copied_img, copied_img_alpha)
                elif m_l_peaks.shape[0] != 0:                    
                    optimal_steer, optimal_velo, showing_img = DynamicWindow_withoutTarget(discrete_steering, curr_velo, curr_steer, m_l_peaks, \
                                                            m_lp_headings, global_navi_info, m_lp_arrows, copied_img, copied_img_alpha)
                else:
                    print("no ray is detected")
                    optimal_steer = 0
                    optimal_velo = 0

                pubImage(pub_image_result, showing_img)
                steer_command.data.append(REAL_VEHICLE_GAIN*optimal_steer)
                pub_dwa_steer.publish(steer_command)
                
                # cv2.imshow("Dynamic Window", imageCall_alpha)
                # cv2.waitKey(1)
            
            
            

if __name__ == '__main__':
    main()
    



# def ray_casting(image, vis = False):
#     SensingRange = 40
#     image[:,:,0][image[:,:,0]<=100.] = 0.
#     image[:,:,1][image[:,:,1]<=100.] = 0.
#     image[:,:,2][image[:,:,2]<=100.] = 0.

#     global histo, fig
#     cX = (int)(image.shape[1]/2)
#     cY = (int)(image.shape[0]/2)+SensingRange #+ 169
#     th_list = []
#     # threshold = 75
#     dis = (int)(180/(DISCRETE_RANGE-1))
#     occupied_draw = []
#     normalized_range = np.zeros(DISCRETE_RANGE)
#     idx = DISCRETE_RANGE-1
#     for th in range(0, 180+1, dis):
#         th_list.append(th-90)
#         currR = 0
#         dR = 0
#         check_done = False
#         while True:
#             dR += 1
#             tX = (int)(cX + dR*math.cos(th*math.pi/180.0))
#             tY = (int)(cY - dR*math.sin(th*math.pi/180.0))
#             # print(tX,tY)
#             if tX >= (int)(imageCall.shape[1]) or tX < 0 or \
#                 tY >= (int)(imageCall.shape[0]) or tY < 0:
#                 break
#             # print(imageCall[tY][tX])        
#             if (not check_done) and list(imageCall[tY][tX]) != [255, 255, 255]:#[255,255,255]-> white(drivable)
#                 occupied_draw.append((tX, tY))
#                 check_done = True
#                 currR = dR
        
#         if (not check_done):
#             occupied_draw.append((tX,tY))
#             currR = dR

#         normalized_range[idx] = currR
#         # normalized_range[idx] = round(currR/dR,4)
#         idx -= 1
    
#     # normalized_range = normalized_range/np.max(normalized_range)
#     # normalized_range = moving_average(normalized_range, MAVG_PARAM)

#     if False:
#     # real-time
#         f_space_distribution = []
#         for th, n_rng in zip(th_list, 100*normalized_range):
#             f_space_distribution += [th]*int(n_rng)
#         density = ss.gaussian_kde(f_space_distribution)
#         density.covariance_factor = lambda : .5# large -> using few component
#         density._compute_covariance()
#         xs = np.linspace(-90,90,300)
#         histo.set_ydata(density(xs))
#         fig.canvas.draw()
#     else:
#         # print(normalized_range.shape)
#         normilized_histo, peaks = inter_detect(normalized_range)
#         histo.set_ydata(normilized_histo)

#         # data = np.array(list(zip(th_list, normalized_range)))
        
#         # f_space_distribution = []
#         # for th, n_rng in zip(th_list, 100*normalized_range):
#         #     f_space_distribution += [th]*int(n_rng)
#         # density = ss.gaussian_kde(f_space_distribution)
#         # density.covariance_factor = lambda : .5# large -> using few component
#         # density._compute_covariance()
#         # xs = np.linspace(-90,90,300)

#         # dpgmm = mixture.BayesianGaussianMixture(n_components = 10, weight_concentration_prior_type='dirichlet_process', max_iter = 20000, verbose = 0).fit(data) # Fitting the DPGMM model
#         # dpgmm = mixture.BayesianGaussianMixture(n_components = 4, weight_concentration_prior_type='dirichlet_process', max_iter = 20000, verbose = 0).fit(density(xs).reshape((-1,1))) # Fitting the DPGMM model
#         # print("active components: %d" % np.sum(dpgmm.weights_ > 0.4))
    
#         print("INTERSECTION: {}".format(True if peaks.shape[0] >= 2 else False))
#         fig.canvas.draw()
    
#     if vis:
#         for occupied_pix in occupied_draw:
#             cv2.line(imageCall, (cX, cY), occupied_pix, (0, 255, 0), 2)

#         cv2.imshow("view", imageCall)
#         cv2.waitKey(1)
# # 