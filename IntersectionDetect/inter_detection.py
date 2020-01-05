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
import re
import weakref

import math

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

histo = None
fig = None
DISCRETE_RANGE = 61
# DISCRETE_RANGE = 31
DEG2RAD = math.pi / 180.0
MAVG_PARAM = 1
imageCall = None

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def image_callback(msg):
    global imageCall
    imageCall = np.fromstring(msg.data, dtype = np.uint8).reshape(96, 96, 3)
    # print(imageCall)

def inter_detect(histogram, distance = 20, height = 0.45):#distance = 20, height = 0.45
    dist_func = histogram/np.max(histogram)
    dist_func = moving_average(dist_func, MAVG_PARAM)
    
    
    local_optima, _ = find_peaks(dist_func, distance = distance, height = (height,))# parameter: distance, height(bound_range of peaks)
    local_optima = 2*(local_optima-45)
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
    draw_lines = []
    normalized_range = np.zeros(DISCRETE_RANGE)
    idx = DISCRETE_RANGE-1

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

        occupied_draw = []
        occupied_draw.append((cX, cY))
        while True:
            dR += 1
            current_yaw = DEG2RAD*steering/steering_ratio
            slip_beta = math.atan2(math.tan(current_yaw), 2)
            dX = real2pix * current_velocity*math.cos(yaw + slip_beta)*dt
            dY = real2pix * current_velocity*math.sin(yaw + slip_beta)*dt
            yaw += dt*current_velocity*math.tan(current_yaw)*math.cos(slip_beta) / wheel_base

            tX = (int)(cX - dR*dY)
            tY = (int)(cY - dR*dX)
            
            ##this line  is for checking the lines overpass the image;
            if tX >= (int)(draw_image.shape[1]) or tX < 0 or \
                tY >= (int)(draw_image.shape[0]) or tY < 0:
                break

            # print(draw_image[tY][tX])
            if (not check_done):
                if list(draw_image[tY][tX]) != [255, 255, 255]: # encounter the occupied pixel(black)
                    check_done = True
                    currR = dR
                    break
                else:
                    occupied_draw.append((tX, tY))

        if (not check_done):
            occupied_draw.append((tX,tY))
            currR = dR
        # if currR == 1:
        #     cv2.imshow("wtf", draw_image)
        #     cv2.waitKey(1)

        # print(curr)
        draw_lines.append(occupied_draw)
        normalized_range[idx] = currR
        # normalized_range[idx] = round(currR/dR,4)
        idx -= 1
    # print(steer_list)
    
    normalized_range = normalized_range/np.max(normalized_range)
    normalized_range = moving_average(normalized_range, MAVG_PARAM)
    histo.set_ydata(normalized_range)
    fig.canvas.draw()


    for line in draw_lines:
        for idx in range(len(line)):
            if idx+1 == len(line):
                break
            else: 
                cv2.line(draw_image, (line[idx][0], line[idx][1]), \
                                     (line[idx+1][0], line[idx+1][1]), (0, 255, 0), 1)

    cv2.imshow("view", draw_image)
    cv2.waitKey(1)



def ray_casting(image, vis = False):
    SensingRange = 40
    image[:,:,0][image[:,:,0]<=100.] = 0.
    image[:,:,1][image[:,:,1]<=100.] = 0.
    image[:,:,2][image[:,:,2]<=100.] = 0.

    global histo, fig
    cX = (int)(image.shape[1]/2)
    cY = (int)(image.shape[0]/2)+SensingRange #+ 169
    th_list = []
    # threshold = 75
    dis = (int)(180/(DISCRETE_RANGE-1))
    occupied_draw = []
    normalized_range = np.zeros(DISCRETE_RANGE)
    idx = DISCRETE_RANGE-1
    for th in range(0, 180+1, dis):
        th_list.append(th-90)
        currR = 0
        dR = 0
        check_done = False
        while True:
            dR += 1
            tX = (int)(cX + dR*math.cos(th*math.pi/180.0))
            tY = (int)(cY - dR*math.sin(th*math.pi/180.0))
            # print(tX,tY)
            if tX >= (int)(imageCall.shape[1]) or tX < 0 or \
                tY >= (int)(imageCall.shape[0]) or tY < 0:
                break
            # print(imageCall[tY][tX])        
            if (not check_done) and list(imageCall[tY][tX]) != [255, 255, 255]:#[255,255,255]-> white(drivable)
                occupied_draw.append((tX, tY))
                check_done = True
                currR = dR
        
        if (not check_done):
            occupied_draw.append((tX,tY))
            currR = dR

        normalized_range[idx] = currR
        # normalized_range[idx] = round(currR/dR,4)
        idx -= 1
    
    # normalized_range = normalized_range/np.max(normalized_range)
    # normalized_range = moving_average(normalized_range, MAVG_PARAM)

    if False:
    # real-time
        f_space_distribution = []
        for th, n_rng in zip(th_list, 100*normalized_range):
            f_space_distribution += [th]*int(n_rng)
        density = ss.gaussian_kde(f_space_distribution)
        density.covariance_factor = lambda : .5# large -> using few component
        density._compute_covariance()
        xs = np.linspace(-90,90,300)
        histo.set_ydata(density(xs))
        fig.canvas.draw()
    else:
        # print(normalized_range.shape)
        normilized_histo, peaks = inter_detect(normalized_range)
        histo.set_ydata(normilized_histo)

        # data = np.array(list(zip(th_list, normalized_range)))
        
        # f_space_distribution = []
        # for th, n_rng in zip(th_list, 100*normalized_range):
        #     f_space_distribution += [th]*int(n_rng)
        # density = ss.gaussian_kde(f_space_distribution)
        # density.covariance_factor = lambda : .5# large -> using few component
        # density._compute_covariance()
        # xs = np.linspace(-90,90,300)

        # dpgmm = mixture.BayesianGaussianMixture(n_components = 10, weight_concentration_prior_type='dirichlet_process', max_iter = 20000, verbose = 0).fit(data) # Fitting the DPGMM model
        # dpgmm = mixture.BayesianGaussianMixture(n_components = 4, weight_concentration_prior_type='dirichlet_process', max_iter = 20000, verbose = 0).fit(density(xs).reshape((-1,1))) # Fitting the DPGMM model
        # print("active components: %d" % np.sum(dpgmm.weights_ > 0.4))
    
        print("INTERSECTION: {}".format(True if peaks.shape[0] >= 2 else False))
        fig.canvas.draw()
    
    if vis:
        for occupied_pix in occupied_draw:
            cv2.line(imageCall, (cX, cY), occupied_pix, (0, 255, 0), 2)

        cv2.imshow("view", imageCall)
        cv2.waitKey(1)

def main():
    
    global fig, histo
    fig, ax = plt.subplots()
    ax.set_title("Intersection")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Histogram")
    
    discrete_steering = SDF(max_steering = 200, k=2)
    histo, = ax.plot(discrete_steering, np.zeros((discrete_steering.shape[0],)), c = 'b', lw=3, alpha=0.5)
    # histo, = ax.plot(np.linspace(-90,90,DISCRETE_RANGE+1-MAVG_PARAM), np.zeros((DISCRETE_RANGE+1-MAVG_PARAM,)), c = 'b', lw=3, alpha=0.5)
    # ax.set_xlim(0, DISCRETE_RANGE)
    ax.set_ylim(0,1.1)
    plt.ion()
    plt.show()

    rospy.init_node('intersection_detect')
    print("Start!")
    sub_image = rospy.Subscriber('front_usb_cam/image_ss', Image, image_callback)    # new_vehicle)
    
    global imageCall
    while not rospy.is_shutdown():
        if imageCall is not None:
            future_trj_casting(imageCall, discrete_steering)
            # ray_casting(imageCall)
            

if __name__ == '__main__':
    main()
    
