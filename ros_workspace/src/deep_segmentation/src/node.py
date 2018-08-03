#!/usr/bin/env python
import math
import cv2
from imutils.video import WebcamVideoStream

import time

from inferer import Inferer
from sector import PCLGen
import perspective
import merger

import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Header
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import PointCloud, Joy, LaserScan

isLaserSafe = True


def current_time(): return int(round(time.time() * 1000))


n = 10000


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def callback1(data):
    global isLaserSafe
    offset = math.radians(90)
    low = int((offset + math.radians(-30) - data.angle_min)/data.angle_increment)
    high = int((offset + math.radians(30) - data.angle_min)/data.angle_increment)
    convolved = np.convolve([data.range_max if math.isinf(dist) else dist for dist in data.ranges[low:high]], [1,1,1,1,1], mode='valid')
    avg = sum(convolved)/len(convolved)
    # print(avg)
    isLaserSafe = not any(dist < 2.0 for dist in convolved)
    # print "lasersafe", isLaserSafe

UNSAFE_POLY = np.array([[75, 180-10], [480-75, 180-10], [480-150, 75], [150, 75]], dtype=np.int32)
UNSAFE_MASK = np.zeros((180, 480), dtype=np.uint8)
UNSAFE_MASK = cv2.fillPoly(UNSAFE_MASK, [UNSAFE_POLY], 255)

def get_pcl(points):
    pcl = PointCloud()
    pcl.header = Header()
    pcl.header.stamp = rospy.Time.now()
    pcl.header.frame_id = "front_camera"
    number_of_pixels = len(points)
    pcl.points = [None] * number_of_pixels
    point_count = 0
    curr = rospy.Time.now()
    xs = []
    for i in xrange(len(points)):
        points[i] = rotate((400, 0), points[i], math.radians(90))
        points[i] = ((0.45)+(points[i][0]/838.926),(-points[i][1]/838.926))
    for p in points:
        pcl.points[point_count] = Point(p[0], p[1], 0)
        point_count += 1
    return pcl

generator = PCLGen((perspective.DST_HEIGHT, perspective.DST_WIDTH), 50)
if __name__ == "__main__":
    import signal
    import sys

    last = False
    lastPCL = None
    rospy.init_node("freespace_stopper")
    pcl_pub = rospy.Publisher("/seg/pcl", PointCloud, queue_size=10)
    scan_sub = rospy.Subscriber("/scan", LaserScan, callback1)
    forward_pub = rospy.Publisher("/forward", String, queue_size=10)
    image_pub = rospy.Publisher("seg/reg", Image, queue_size=2)
    bridge = CvBridge()

    inferer = Inferer(2502)

    def end(sig, frame):
        print "\n\nClosing TF sessions"
        inferer.sess.close()
        print "done."
        sys.exit(0)
    signal.signal(signal.SIGINT, end)

    cam = WebcamVideoStream(src=1).start()
    while True:
        frame = cam.read()
        start = current_time()
        frame = perspective.undistort_internal(frame)
        frame = frame[240:, :]
        frame = cv2.resize(frame, (480, 180))
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        unsafe = inferer.infer(frame)
        unsafe_top = perspective.undistorted_to_top_down(unsafe)
        unsafe = cv2.erode(unsafe, (5,5), iterations=6)
        unsafe = cv2.dilate(unsafe, (5,5), iterations=6)
        #print(unsafe.shape, UNSAFE_MASK.shape)
        unsafe = cv2.bitwise_and(UNSAFE_MASK, unsafe)
        numPx = cv2.countNonZero(unsafe)
        print numPx

	THRESHOLD = 1200
        if numPx < THRESHOLD and isLaserSafe:
            forward_pub.publish("1")
        elif numPx < THRESHOLD and not isLaserSafe:
            forward_pub.publish("laser")
        elif numPx >= THRESHOLD and isLaserSafe:
            forward_pub.publish("infer")
        else:
            forward_pub.publish("both")


        curr = rospy.Time.now()

        regular = cv2.addWeighted(frame, 0.7, cv2.cvtColor(
             unsafe, cv2.COLOR_GRAY2RGB), 0.3, 0.0)


        cv2.polylines(regular, np.array([UNSAFE_POLY], dtype=np.int32), True, (255, 0, 0))

        try:
            image_pub.publish(bridge.cv2_to_imgmsg(regular, encoding="bgr8"))
        except CvBridgeError as e:
            print e
        time_pub_image = current_time()
        points = generator.find_points(unsafe_top)
        pcl_pub.publish(get_pcl(points))
        isLaserSafe = True
        # print "fps:" + str(1000.0/(time_pub_image-start))
