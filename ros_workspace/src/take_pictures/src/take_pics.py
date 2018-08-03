#!/usr/bin/env python
from __future__ import print_function
import cv2

import rospy
from sensor_msgs.msg import Joy

last = False
cap = cv2.VideoCapture(1)
n = 10000

def callback(data):
	global last
	state = data.buttons[6] == 1 or data.buttons[7] == 1
	trigger = (not state) and last
	last = state

	if trigger:
		global cap
		global n
		img = cap.read()[1]
		n = n + 1
		cv2.imwrite("/home/nvidia/data/img_"+str(n)+".png", img)

rospy.Subscriber("/bluetooth_teleop/joy", Joy, callback)
rospy.init_node("take_pics")
rospy.spin()
