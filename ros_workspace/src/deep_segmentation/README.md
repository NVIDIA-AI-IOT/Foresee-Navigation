# Deep Semantic Segmentation

## What is Deep Semantic Segmentation?

In Computer Vision applications of Deep Learning, there are three main tasks that Convolutional Neural Networks are extremely good at: Image Classification, Object Detection, and Semantic Segmentation. While image classification assigns one class to an entire image and object detection draws a box around distinct objects in an image, semantic segmentation assigns a class to each pixel in an image.

![Image of PASCAL VOC Semantic Segmentation](https://meetshah1995.github.io/images/projects/semseg/semseg.png)

*Example of Pascal VOC Semantic Segmentation*

## Why do we want to use Semantic Segmentation?

Well, the main idea behind this project is that 2D LiDAR or other Laser Range Finders cannot identify things like glass, railing, or stairs as per their limitations to one-dimension and because of the way light (including lasers) reflects off of glass. We need a way to detect whether certain places in front of the robot are safe to move to or not, especially when things such as glass, railing, or stairs exist in a robot's path. Since the precision required to move a robot in real-time around these objects can only come with getting as much information from the camera as possible, we decide to create a **freespace detector** which can label each pixel in an image as being freespace or occupied (by any obstacles: walls, glass, railing, stairs, moving objects, etc.)

## Our Segmentation Model

We chose to use DeepLab_V3 as our semantic segmentation model because it made good tradeoffs between accuracy and speed. After all, we're deploying this on a mobile robot with a Jetson TX2 so we need to make these tradeoffs. While we did try using other models such as ENet and SegNet for semantic segmentation, none of them really matched up with DeepLab_v3's ability to eliminate random noise while still being able to function at its 3 FPS speed with everything else running on the robot.

As visualized in TensorBoard, the model architecture is displayed here.

![Image of DeepLab V3 Architecture in TensorBoard](https://i.imgur.com/PVh6Grr.png)

Learn more about DeepLab by reading the paper here: https://arxiv.org/pdf/1706.05587.pdf

## Training our Model

We trained the model using TensorFlow in Python on a Tesla V100 but you should be able to train this model on any GPU or CPU. Please view the main README to find other software required to train this model. After you download all required software, you can follow the steps below.

### Preparing the Data

First, you must collect data to or find a dataset that you want to train this freespace detector on. While the end goal of the model is to have it detect whether certain spaces are either free or not, we chose to use more than two classes to train the model. Depending on the space you plan to navigate around, choose distinct visual classes that you'd like to use as the different classes you model learns. For example, in the NVIDIA Endeavor building, we found three different types of floor and 1 type of stairs. This yields 5 classes as per, 3 types of floor, 1 type of stairs, and background (anything that isn't these other classes). This seems to help the model train better rather than just having two classes because now it's able to separate different types of objects/textures from each other. We can then deem that any type of floor detected is safe for the robot to move on and that any stairs or background is unsafe for the robot to move on.

If you choose to collect data, make sure that you first decide upon a uniform image resolution. For our case, we chose to take an image from a 480 x 640 camera, scale it to 360 x 480 and finally cut the top half of the image off which yielded a 180 x 480 image input resolution. After we collected data in this way, we labeled it using the SXLT labeling tool we created which is in the base folder of this repository. Since the labels of a semantic segmentation model are image masks, we chose to label image classes by pixel value. All class 1 labels would be of pixel value 1 and so on. After doing this, we converted the labels to one-channel images based on class. **Please make sure that you do this because the training script won't work unless the image labels/masks are one-channel images.** Finally, we applied various transformations to the dataset such as sheers, rotations, vertical flips, etc but this all depends on your specific scenario so we haven't included a script that automatically does this. 

Now that you have an images and a labels folder, we need to input these into our training script. To do this, we use something called a TensorFlow TFRecord. This is essentially just an easier way to store a dataset. There are many scripts online that can convert your dataset to the TFRecords format so after you do this, you are ready to go!

### Running the Training Script

The script to run in order to start training the DeepLab model is `train.py`.

If you have your dataset in the following folder structure, you are good to just do `python train.py` to run the code with default parameters:

```
LOG_FOLDER = './tboard_logs'
TRAIN_DATASET_DIR="./dataset/tfrecords"
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
```

Otherwise, please either change these variables by editing the top of `train.py` or move and rename you folders and files to be in this structure.

There are many different parameters you can change when training the DeepLab model and if you'd like to run it with the parameters that we used, you may just use `python train.py` to run the model. However, if you'd like to change these parameters, you have many different options for your command line arguments.

Argument Description | Argument Syntax | Default Value
------------ | ------------- | -------------
batch norm epsilon argument for batch normalization | batch_norm_epsilon | 1e-5
batch norm decay argument for batch normalization | batch_norm_decay | 0.9997
Number of classes to be predicted | number_of_classes | 5
l2 regularizer parameter | l2_regularizer | 0.0001
initial learning rate | starting_learning_rate | 0.00001
Spatial Pyramid Pooling rates | multi_grid | [1,2,4]
Id of the GPU to be used | gpu_id | 0
Image Cropsize | crop_size | 480
Resnet model to use as feature extractor | resnet_model | resnet_v2_50
Best validation loss value | current_best_val_loss | 99999
Accumulated validation intersection over union | accumulated_validation_miou | 0
Batch size for network train | batch_size | 12

After running the training script, you may visualize the loss of the model while training using `tensorboard --logdir=tboard_logs` if you're in the `deep_segmentation/src` directory.

## Using the Model to Navigate

If you're reading this step, it means that you're satisfied with the output you model is giving. You can test your model on data by simply creating a TensorFlow Session and passing your data through a Placeholder. If you don't know how to do this, you may view `inferer.py`.

Okay, cool. Now here's the deal. Although you just created this model that can detect what's freespace and what isn't, you still can't really use that to navigate your robot. What you've got to do is convert this mask data that your model returns into data that fits under the robot's coordinate system. Since we want to eventually plug this data into a ROS NavStack, what we really need is a PointCloud or a LaserScan output from the model.

The first step in doing this is to put the image onto a coordinate plane using a top-down image tranform which transforms the image mask into something that can viewed on a top-down plane. To do this, we use `perspective.py` which gets an OpenCV transform matrix based on various tuned constants based on your camera and camera height. Please read through the following links to understand how this works and how you might want to tune this. The values in the file are based on our Jackal Robot configuration mentioned in the main README.

https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

After we get a top-down transform of the image, we are able to use it to find the obstacles nearest or furthest from us as a cloud of points. We do this using `sector.py`, an algorithm which we designed that divides the image in `n=50` masks and finds the nearest contour on those masks using OpenCV and finally returning a set of coordinate points relative to the camera.