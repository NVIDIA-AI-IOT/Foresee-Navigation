#!/usr/bin/env python
from __future__ import print_function
import sys
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3
import time

def println(*data):
	print(*data, file=sys.stderr)

def getTime():
	return int(round(time.time() * 1000))

stop = Twist(linear=Vector3(0.0, 0.0, 0.0))
lastTime = getTime()
lastEnableTime = 0
message = stop
enabled = False
canGoForwardCount = 0
lastForwardTime = getTime()

def callback(data):
	global message
	global lastTime
	message = data
	# Nothing to see here
	# message.angular.z = -1.0 * message.angular.z
	lastTime = getTime()

def enabledCallback(data):
	global enabled
	global lastEnableTime
	if "1" in data.data:
		enabled = True
	else:
		enabled = False
	lastEnableTime = getTime()

def forwardCallback(data):
	global canGoForwardCount
	global lastForwardTime
	lastForwardTime = getTime()
	if "1" in data.data:
		canGoForwardCount = min(canGoForwardCount + 1, 5)
	else:
		canGoForwardCount = max(canGoForwardCount - 2, 0)


rospy.init_node("watchdog", anonymous=True)
velpub = rospy.Publisher("/jackal_velocity_controller/cmd_vel", Twist, queue_size=10)
velsub = rospy.Subscriber("/cmd_vel", Twist, callback)
enablesub = rospy.Subscriber("/enabled", String, enabledCallback)
forwardsub = rospy.Subscriber("/forward", String, forwardCallback)

while not rospy.is_shutdown():
	print("count: ", canGoForwardCount)
	if getTime() - lastForwardTime > 350 and message.linear.x > 0:
		message.linear.x = 0
	elif canGoForwardCount <= 2 and message.linear.x > 0:
		message.linear.x = canGoForwardCount * message.linear.x / 4.0

	if getTime() - lastTime > 250 or not enabled or getTime() - lastEnableTime > 100:
		print("STOPPING DUE TO TIMEOUT")
		velpub.publish(stop)
	else:
		velpub.publish(message)
