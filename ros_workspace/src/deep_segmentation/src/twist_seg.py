print "begin imports"
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import network
slim = tf.contrib.slim
import os
import argparse
import json
from read_data import tf_record_parser, scale_image_with_crop_padding
import training
from metrics import *
import cv2
from imutils.video import WebcamVideoStream
import perspective

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Twist
print "end imports"

model_name = str(14047)

log_folder = './tboard_logs'

percent = 34

'''def adjust_percent(x):
    global percent
    percent = x

cv2.namedWindow("controls")
cv2.createTrackbar("percent", "controls", percent, 100, adjust_percent)'''


with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)
tf.set_random_seed(1)
holder = tf.placeholder(dtype=tf.float32, shape=[1, 180, 480, 3])
logits_tf =  network.deeplab_v3(holder, args, is_training=False, reuse=False)


predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def get_twist_left():
    out = Twist()
    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = -0.8
    return out

def get_twist_right():
    out = Twist()
    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0.8
    return out

def get_twist_forward():
    out = Twist()
    out.linear.x = 0.4
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0
    return out

def get_twist_stop():
    out = Twist()
    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0
    return out
#def signal_handler()

UNSAFE_POLY = np.array([[75, 180-15], [450, 180-15], [370, 100], [150, 100]], dtype=np.int32)
UNSAFE_MASK = np.zeros((180, 480), dtype=np.uint8)
UNSAFE_MASK = cv2.fillPoly(UNSAFE_MASK, [UNSAFE_POLY], 255)
#cv2.imshow("poly", UNSAFE_MASK)
#cv2.waitKey(0)

def main():
    import signal
    import sys
    
    rospy.init_node("obstacle_node")
    forward_pub = rospy.Publisher("/forward", String, queue_size=10)
    #data_pub = rospy.Publisher('enabled_test', String, queue_size=10)
    #velocity_pub = rospy.Publisher('cmd_vel_test', Twist, queue_size=10)
    image_pub = rospy.Publisher("img/reg", Image)
    #image_pub_dis = rospy.Publisher("img/td", Image)
    bridge = CvBridge()
    sess = tf.Session(config=config)
    
    def end(sig, frame):
        print "\n\nClosing TF sessions"
        sess.close()
        print "done."
        sys.exit(0)
    
    signal.signal(signal.SIGINT, end)

    new_trunc = tf.constant(5, dtype=tf.float32, shape=[7, 7, 3, 64])
    tf.import_graph_def(tf.get_default_graph().as_graph_def(), input_map={"resnet_v2_50/conv1/weights/Initializer/truncated_normal/TruncatedNormal:0": new_trunc})
    saver = tf.train.Saver()
    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    test_folder = os.path.join(log_folder, model_name, "test")
    train_folder = os.path.join(log_folder, model_name, "train")
    # Restore variables from disk.
    saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
    print("Model", model_name, "restored.")
    cam = WebcamVideoStream(src=1).start()
    print "finna run"
    import time
    last_time = (int(round(time.time() * 1000)))
    while True:
		last_time = (int(round(time.time() * 1000)))
		frame = cam.read()
		#frame = perspective.undistort_internal(frame)
		#cv2.imshow('yo1', frame)
		frame = frame[240:, :]
		frame = cv2.resize(frame, (480, 180))
		frame = cv2.GaussianBlur(frame, (3, 3), 0)
		out = sess.run(predictions_tf, feed_dict={holder: [frame]})
		out = np.squeeze(out)
		bg = cv2.inRange(out, 0, 0)
		stairs = cv2.inRange(out, 3, 255)
		unsafe = cv2.bitwise_or(stairs, bg)
		#cv2.imshow("rawm", unsafe)
		unsafe = cv2.erode(unsafe, (5,5), iterations=6)
		unsafe = cv2.dilate(unsafe, (5,5), iterations=6)
		unsafe = cv2.bitwise_and(UNSAFE_MASK, unsafe)
		#cv2.imshow("asdkj", unsafe)
		numPx = cv2.countNonZero(unsafe)
		print numPx
		if numPx > 450:
			forward_pub.publish("0")
		else:
			forward_pub.publish("1")
			
		regular = cv2.addWeighted(frame, 0.7, cv2.cvtColor(unsafe, cv2.COLOR_GRAY2RGB), 0.3, 0.0)
		
		#distort_unsafe = perspective.undistort_to_top_down(unsafe)
		#distort_frame = perspective.undistort_to_top_down(frame)
		#distorted = cv2.addWeighted(distort_frame, 0.7, cv2.cvtColor(distort_unsafe, cv2.COLOR_GRAY2RGB), 0.3, 0.0)
		#cv2.imshow('yo', distort_unsafe)
		try:
			image_pub.publish(bridge.cv2_to_imgmsg(regular, encoding="bgr8"))
		#	image_pub_dis.publish(bridge.cv2_to_imgmsg(distorted, encoding="bgr8"))
		except CvBridgeError as e: print e
		
		#unsafe1 = unsafe[90:180, 0:160]
		#unsafe2 = unsafe[90:180, 160:320]
		#unsafe3 = unsafe[90:180, 320:480]
		#unsafe = unsafe[90:180, 100:380]
		#num1 = cv2.countNonZero(unsafe1)
		#num2 = cv2.countNonZero(unsafe2)
		#num3 = cv2.countNonZero(unsafe3)
		#num = cv2.countNonZero(unsafe)
            
		#if num < (percent*90*280/100.0):
		#	if num1 <= num2 and num1 <= num3: velocity_pub.publish(get_twist_left())
		#	elif num2 <= num3 and num2 <= num1: velocity_pub.publish(get_twist_forward())
		#	else: velocity_pub.publish(get_twist_right())
		#	data_pub.publish('go')
		#else:
		#	data_pub.publish('stop')
		#	velocity_pub.publish(get_twist_stop())
		#	
		print (last_time - (int(round(time.time() * 1000))))
		#cv2.imshow('yo', regular)
		#cv2.waitKey(1)

if __name__ == "__main__":
    main()
