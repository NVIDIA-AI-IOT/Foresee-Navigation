import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from matplotlib import pyplot as plt
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
plt.interactive(False)

model_name = str(8628)

log_folder = './tboard_logs'

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)
tf.set_random_seed(1)
holder = tf.placeholder(dtype=tf.float32, shape=[1, 180, 299, 3])
logits_tf =  network.deeplab_v3(holder, args, is_training=False, reuse=False)


predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
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

    while True:
        frame = cam.read()
        frame = frame[0:180, 91:390]
        out = sess.run(predictions_tf, feed_dict={holder: [frame]})
        out = np.squeeze(out)
        bg = cv2.inRange(out, 0, 0)
        stairs = cv2.inRange(out, 3, 255)
        unsafe = cv2.bitwise_or(stairs, bg)
        cv2.imshow("frame", frame)
        cv2.imshow("out", unsafe)
        cv2.waitKey(1)
