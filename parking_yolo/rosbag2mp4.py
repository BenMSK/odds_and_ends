import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import math
import random
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class Bag2Video:
   
    def __init__(self):
        self.avm_img_sub = rospy.Subscriber("avm/image_raw", Image, self.image_callback)
        # self.stop = rospy.Subscriber("stop", Bool, self.stopit)
        self.image = None
        self.good = False
        self.out = cv2.VideoWriter(
                "test190925_2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                (640, 480))

    def image_callback(self, msg):
        self.image = np.fromstring(msg.data, dtype = np.uint8).reshape(480, 640, 3) 
        self.out.write(self.image)
        
def main():
    hoho = Bag2Video()
    rospy.init_node('yolo_v3')
    # YOLO()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        hoho.out.relaese()
        print("Shutting down")
    

if __name__ == "__main__":
    main()
    
