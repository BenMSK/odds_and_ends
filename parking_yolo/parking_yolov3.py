from ctypes import *

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
# from cv_bridge import CvBridge, CvBridgeError
import math
import random
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
import darknet

class YOLOv3:
   
    def __init__(self):
        self.avm_img_sub = rospy.Subscriber("avm_usb_cam/image_raw", Image, self.image_callback)
        self.parking_cand_pub = rospy.Publisher('/parking_cands', PoseArray, queue_size=1)
        
        self.metaMain = None
        self.netMain = None
        self.altNames = None
        self.image = None
        configPath = "./yolo-obj_onlyFree.cfg"
        weightPath = "./yolo-obj_parking.weights"#ms: loading the weight file
        # weightPath = "./yolo-obj_low_dimension.weights"#ms: the size of input has to be (224,224)
        metaPath = "./obj.data"

        if self.netMain is None:
            self.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                darknet.network_height(self.netMain),3)

    def image_callback(self, msg):
        self.image = np.fromstring(msg.data, dtype = np.uint8).reshape(320, 480, 3)#avm
        # self.image = np.fromstring(msg.data, dtype = np.uint8).reshape(480, 640, 3)#front
        prev_time = time.time()
    
    #========================================================================================================msmsmsms
        frame_resized = cv2.resize(self.image,
                                    (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(self.darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)

        cands = PoseArray()# [x, y, heading, velocity]
        for detection in detections:
            temp = Pose()
            temp.position.x = detection[2][0]
            temp.position.y = detection[2][1]
            temp.position.z = detection[1]
            cands.poses.append(temp)
        self.parking_cand_pub.publish(cands)
    # #========================================================================================================                      
        # print("frame: ", 1/(time.time()-prev_time))

        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Demo', image)
        cv2.waitKey(3)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        # print(detection[0].decode("utf-8")=='occupied')
        if detection[0].decode("utf-8") == 'occupied':
            # print("wow")
            continue
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def main():
    yolo = YOLOv3()
    rospy.init_node('yolo_v3')
    # YOLO()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
    
