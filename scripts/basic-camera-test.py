import numpy as np
import cv2
import argparse
import sys

# Ros libraries
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

parser = argparse.ArgumentParser()
parser.add_argument('--cam', type=int, default=0)
parser.add_argument('--hd', action='store_true', help='Save in 720p if possible')
args = parser.parse_args()

class image_stream:

    def __init__(self):
        self.bridge = CvBridge()
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/image_raw", Image, self.callback,  queue_size = 1)

    def callback(self, ros_data):
        cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")

        cv2.imshow('frame',cv_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            sys.exit(0)


def main(args):
    '''Initializes and cleanup ros node'''
    istream = image_stream()
    rospy.init_node('image_stream', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
        cv2.destroyAllWindows()
  
if __name__ == '__main__':
    main(sys.argv)
