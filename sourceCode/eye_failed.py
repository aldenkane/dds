# eye.py
# Production Code for OptoSwim Eye Module
# Property of OptoSwim
#   Author: Alden Kane

import cv2
import numpy as np
import math
import time
from opto import dds_yolo

#######################################################
# Coding Conventions
#######################################################
# All variables transported off of module are UPPER_CASE
# Algorithms that take more than 10 lines are turned into functions
# All functions housed in opto.py

#######################################################
# Initiate Camera, Declare Globals
#######################################################
time.sleep(5)                                           # Camera Warmup For 5s
#cam = cv2.VideoCapture(0)                               # Initiate Camera Stream
cam = cv2.VideoCapture('../dataSet/swim3/swim3.1-12-of-14.mp4')                 # PROTOTYPING
first_frame = None                                      # Set First Frame for Motion Detection
res_scale = 0.5                                         # Resize Info
frames_processed = 0                                    # Iterator for Frames Processed
sample_rate = 30                                        # Sample Yolo Every N Frames

JSON_FILE_PATH = '../last_Image/event.json'             # JSON Writing Globals
SERIAL_NO = '0001'
SWIMMER_DETECTED = False
DROWNING_DETECT = False
NUMBER_SWIMMERS = 0

#######################################################
# YOLO Initialization
#######################################################
# Load YOLOv3 Retrained Model
net = cv2.dnn.readNet("../yolo/yolov3_custom_train_final.weights", "../yolo/yolov3_custom_train.cfg")
classes = []
with open("../yolo/yolo.names", "r") as f:              # Get class names
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()                       # Get the output layers of our YOLO model
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while (True):
    starting_time = time.time()                         # Starting time for FPS
    retval, frame = cam.read()                          # Camera read

    if frames_processed % sample_rate == 1:
        frame_yolo, yolo_swimmer_count = dds_yolo(frame, net, output_layers, classes)

    frames_processed += 1                               # Iterate on Frames Processed

