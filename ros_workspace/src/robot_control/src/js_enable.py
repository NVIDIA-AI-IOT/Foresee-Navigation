#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import time

def println(*data):
	print(*data, file=sys.stderr)

def getTime():
	return int(round(time.time() * 1000))

message = False

def callback(data):
	global message
	global lastTime
	message = (1 in data.buttons) 
	lastTime = getTime()
	if message:
		pub.publish(String(data="1"))

rospy.init_node("js_enable", anonymous=True)
pub = rospy.Publisher("/enabled", String, queue_size=10)
rospy.Subscriber("/joy", Joy, callback)
rospy.spin()
