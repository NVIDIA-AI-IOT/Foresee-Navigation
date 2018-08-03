#!/usr/bin/env python
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import network
slim = tf.contrib.slim
import os
import json
import cv2
import signal
import sys

class Inferer:

    def __init__(self, model_num):
        model_name = str(model_num)
        log_folder = '/home/nvidia/wilcove/ros_workspace/src/deep_segmentation/src/tboard_logs'
        with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
            args = json.load(fp)
        class Dotdict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__
        args = Dotdict(args)
        tf.set_random_seed(1)
        self.holder = tf.placeholder(dtype=tf.float32, shape=[1, 180, 480, 3])
        logits_tf =  network.deeplab_v3(self.holder, args, is_training=False, reuse=False)
        self.predictions_tf = tf.argmax(logits_tf, axis=3)
        probabilities_tf = tf.nn.softmax(logits_tf)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        test_folder = os.path.join(log_folder, model_name, "test")
        train_folder = os.path.join(log_folder, model_name, "train")
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(train_folder, "model.ckpt"))
        print("Model", model_name, "restored.")

    def infer(self, image):   
        out = self.sess.run(self.predictions_tf, feed_dict={self.holder: [image]})
        out = np.squeeze(out)
        bg = cv2.inRange(out, 0, 0)
        stairs = cv2.inRange(out, 3, 255)
        unsafe = cv2.bitwise_or(stairs, bg)

        return unsafe
		
'''if __name__ == "__main__":
    inferer = Inferer()
    cam = cv2.VideoCapture(1)
    ret, frame = cam.read()
    frame = frame[240:, :]
    frame = cv2.resize(frame, (480, 180))
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    out = inferer.infer(frame)
    cv2.imshow('yo', out)
    cv2.waitKey(0)'''
