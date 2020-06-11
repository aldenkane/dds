# uw.py
# OptoSwim embedded module for use in pools
# Uses .json configuration file for loading environment parameters
#   Author: Alden Kane

import cv2
import numpy as np
import math
import time
import json
import logging
from ez_cv import do_canny, segment_for_bottom, find_bottom_line, generate_bottom_mask, write_pool_info_json, check_water_quality

########################################
# Section 1: Initialize Globals, Video Parameters, Windows, Load JSON File
##########################################

conf = json.load(open('./conf.json'))               # Open .json Config File
device = json.load(open('./device.json'))              # Device Parameters
first_frame = None                                  # Motion Detection First Frame
avg = None                                          # Motion Detection Averaging Frame
bsmog = cv2.bgsegm.createBackgroundSubtractorMOG(history=150, nmixtures=5, backgroundRatio=0.1, noiseSigma=0)  # Background Subtractor
debounce_timer = 0                                  # Less Oscillation in Boxing
frames_processed = 0                                # Iterate on Frames Processed
starting_time = time.time()                         # For Measuring FPS
timer = 0.00                                        # For Debouncing Boxes

drowningRisk = 0
FPS = 30

JSON_FILE_PATH = '../last_Image/event.json'         # VARIABLES THAT ARE WRITTEN TO JSON FILE
NUMBER_SWIMMERS = 0
SWIMMER_DETECTED = False
DROWNING_DETECT = False
SERIAL_NO = device["serial_no"]

if conf["raspberry_pi"]:                            # Initiate logging for Raspberry Pi
    time.sleep(int(conf["camera_warmup_time"]))     # Camera Warmup
    time_tuple = time.localtime()                   # Logging Time
    log_Filename = '../logs/' + str(time_tuple[1]) + '.' + str(time_tuple[2]) + '.' + str(time_tuple[0]) + '_' + str(
        time_tuple[3]) + '.' + str(time_tuple[4]) + '.' + str(time_tuple[5]) + '_eye_V0.1.log'
    logging.basicConfig(filename=str(log_Filename), level=logging.DEBUG)
    logging.debug('Accessed Log File')

cam = cv2.VideoCapture(0)                           # Webcam Capture

#######################################################
# YOLO Initialization
#######################################################
net = cv2.dnn.readNet("../yolo/yolov3_custom_train_final.weights", "../yolo/yolov3_custom_train.cfg")
sample_rate_yolo = 30                                        # How often to run YOLO
classes = []                                            # Class list
with open("../yolo/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layers of our YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Font and random colors useful later when displaying the results
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initiate Windows Based on show_video in conf.json
if conf["show_video"]:
    cv2.namedWindow("Color Detection: Binary image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color Detection: Binary image", 400, 225)

    cv2.namedWindow("Color Detection: Image after Morphological Operations", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color Detection: Image after Morphological Operations", 400, 225)

    cv2.namedWindow("FF Motion Detection: Binary Image after Morphological Operations", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FF Motion Detection: Binary Image after Morphological Operations", 400, 225)

    cv2.namedWindow("Motion Detection: Absolute Difference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Detection: Absolute Difference", 400, 225)

    cv2.namedWindow("BSMOG Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BSMOG Mask", 400, 225)

    cv2.namedWindow("Logical AND'ing of Motion and Color Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Logical AND'ing of Motion and Color Contours", 400, 225)

    cv2.namedWindow("DDS: Underwater Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DDS: Underwater Video Feed", 400, 225)

    cv2.namedWindow("YOLOv3 Boxing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv3 Boxing", 400, 225)

    cv2.namedWindow("Video + Environment Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video + Environment Parameters", 400, 225)

    cv2.namedWindow("Averaging Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Averaging Motion Detection", 400, 225)

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while True:
    #######################################################
    # Pre-Processing - Initialize Window Position, Set Timers, Declare Kernels
    #######################################################
    if conf["show_video"]:
        cv2.moveWindow("Color Detection: Binary image", 840, 0)
        cv2.moveWindow("Color Detection: Image after Morphological Operations", 0, 300)
        cv2.moveWindow("Motion Detection: Absolute Difference", 840, 300)
        cv2.moveWindow("BSMOG Mask", 420, 300)
        cv2.moveWindow("Logical AND'ing of Motion and Color Contours", 420, 0)
        cv2.moveWindow("FF Motion Detection: Binary Image after Morphological Operations", 0, 600)
        cv2.moveWindow("YOLOv3 Boxing", 420, 600)
        cv2.moveWindow("Video + Environment Parameters", 840, 600)
        cv2.moveWindow("Averaging Motion Detection", 1260, 0)
        cv2.moveWindow("DDS: Underwater Video Feed", 0, 0)

    # Declare kernels of various sizes for experimentation
    kernel_2 = np.ones((2, 2), np.uint8)
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_4 = np.ones((4, 4), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_6 = np.ones((6, 6), np.uint8)
    kernel_7 = np.ones((7, 7), np.uint8)
    kernel_8 = np.ones((8, 8), np.uint8)
    kernel_9 = np.ones((9, 9), np.uint8)
    kernel_10 = np.ones((10, 10), np.uint8)
    kernel_11 = np.ones((11, 11), np.uint8)
    kernel_12 = np.ones((12, 12), np.uint8)
    kernel_13 = np.ones((13, 13), np.uint8)
    kernel_14 = np.ones((14, 14), np.uint8)
    kernel_15 = np.ones((15, 15), np.uint8)
    kernel_21 = np.ones((21, 21), np.uint8)
    kernel_40 = np.ones((40, 40), np.uint8)
    kernel_71 = np.ones((71, 71), np.uint8)

    #######################################################
    # Read Image, Ch
    #######################################################
    retval, frame = cam.read()                              # Image for main channel
    img_yolo = frame.copy()                                 # Image for yolo stream

    var_of_laplacian = check_water_quality(frame)           # Check image sharpness using variance of Laplacian

    #######################################################
    # Section 2: Color Detection - Set up HSV Color Detection Bounds
    #######################################################
    # Declare hsv upper and lower bounds for color image detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([84, 0, 0])
    upper = np.array([111, 255, 200])
    binary_image = cv2.inRange(hsv, lower, upper)

    # Morphological operations for cleaner image
    # Remove artifacts of smaller pool lines, movement of water with initial erosion
    binary_image = cv2.erode(binary_image, kernel_3)
    # Fill in binary image of swimmers with closing
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=kernel_21)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=kernel_21)
    # Erosion to clear up final image
    binary_image = cv2.erode(binary_image, kernel_5)
    # Dilation to Connect Swim Suits - This adds noise to this binary image, but this is filtered out by logical AND later
    binary_image = cv2.dilate(binary_image, kernel_21, iterations=2)

    #######################################################
    # Section 3: Motion Detection - Grayscale Image Processing, Absolute Differencing for Motion Detection, Thresholding
    #######################################################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (conf["gaussian_blur_size"], conf["gaussian_blur_size"]), 0)

    if first_frame is None:     # Initialize First Frame
        first_frame = gray
        continue

    if avg is None:             # Initialize Average Frame
        avg = gray.copy().astype("float")
        continue

    # First Frame Motion Detection
    delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(delta, conf["first_frame_threshold"], 255, cv2.THRESH_BINARY)[1]   # Used to be 10

    # Averaging Motion Detection
    cv2.accumulateWeighted(gray, avg, conf["avg_alpha"])
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh_avg = cv2.threshold(frameDelta, conf["avg_delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh_avg = cv2.morphologyEx(thresh_avg, cv2.MORPH_OPEN, kernel_7)

    # BSMOG Motion Detection w/ Morphological Operations
    bs_mask = bsmog.apply(frame)
    bs_mask = cv2.erode(bs_mask, kernel_3, iterations=3)
    bs_mask = cv2.dilate(bs_mask, kernel_9, iterations=3)

    # First Frame Motion Detection Morphological Operations
    thresh = cv2.erode(thresh, kernel_3, iterations=2)
    # thresh = cv2.erode(thresh, kernel_7, iterations=1)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel_21)
    # thresh = cv2.dilate(thresh, kernel_11, iterations=5)
    # thresh = cv2.dilate(thresh, kernel_13, iterations=4)

    #######################################################
    # Section 7: Perform Logical AND'ing of Binary Image, Implement Size Based Object Detection
    #######################################################
    # Perform bitwise AND of motion AND color contours. Change 'thresh' to preferred color motion detection
    binary_intersection = cv2.bitwise_and(bs_mask, binary_image)

    # Find contours
    contours, hierarchy = cv2.findContours(binary_intersection,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Ignore bounding boxes smaller than "minObjectSize". Tuned for optimal swimmer detection
    min_object_size = 80

    #######################################################
    # Section 9: Object Detection and Localization w/ Drowning Detection Feature Built In
    #######################################################
    swimmers = []                                                 # Make list of boxes
    if contours:
        # Detect all swimmers, i.e. all objects with contours
        for contours in contours:
            # use just the first contour to draw a rectangle
            x, y, w, h = cv2.boundingRect(contours)
            if w > min_object_size or h > min_object_size:        # If statement to filter out small objects
                swimmers.append('s')                              # Mask for Number of Swimmers
                NUMBER_SWIMMERS = len(swimmers)                   # Gives number of swimmers in pool
                SWIMMER_DETECTED = True                           # This designates a swimmer detection
                timer += 1                                        # Iterate on My Time
                scaled_T = math.ceil(timer / FPS)                 # Scaled Time that Accounts for FPS

                # Put up bounding boxes w/ Text, If Statement for Timing
                if not drowningRisk:
                    debounce_timer = (debounce_timer + 1) / FPS
                    if debounce_timer < 0.1:
                        timer += 1
                        scaled_T = math.ceil(timer / FPS)
                    elif debounce_timer > 1:
                        timer = 0
                        drowningRisk = 0
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame,  # image
                                "Swimmer(s)",  # text
                                (x, y - 10),  # start position
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # size
                                (0, 255, 0),  # BGR color
                                1,  # thickness
                                cv2.LINE_AA)  # type of line
                    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

        #######################################################
        # YOLO
        #######################################################
        if frames_processed % sample_rate_yolo == 1:
            height, width, channels = img_yolo.shape              # Get img shape for CV2 Blob
            # Normalize input frame using blobFromImage, SwapRB Codes, and Scale Value to 1/255
            blob = cv2.dnn.blobFromImage(img_yolo, scalefactor=1 / 255, size=(320, 320), mean=0, swapRB=True, crop=False)
            net.setInput(blob)                                    # Set input of the net
            outputs = net.forward(output_layers)                  # Predict outputs using net.forward
            class_ids = []                                        # Initialize lists for displaying results, now that detection is done
            confidences = []
            boxes = []

            for out in outputs:
                for detection in out:
                    # Get scores of detection
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Now, have class ID's and have detections. Now, ignore, scores of low confidence. Get bounding box coordinates here
                    if confidence >= 0.2:
                        # Multiply this by width and height
                        x = width * detection[0]    # Corresponds to X center
                        y = height * detection[1]   # Corresponds to Y center
                        w = width * detection[2]    # Corresponds to the box width, and
                        h = height * detection[3]   # Corresponds to the box height

                        # Append info to boxes list (List w/ in list)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Use cv2.dnn.NMS Boxes and play w/ NMS Threshold, Score Threshold, and top_k for detections. Top_k controls how many boxes/swimmers can be detected
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.2, eta=1, top_k=10)

            # Now, display stuff for YOLO
            for i in range(len(boxes)):
                if i in indices:
                    for num_detects in indices:
                        x, y, w, h = boxes[i]
                        x1 = int(x - (w / 2))
                        y1 = int(y - (h / 2))
                        x2 = int(x + (w / 2))
                        y2 = int(y + (h / 2))
                        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), colors[class_ids[i]], 3)
                        cv2.putText(img_yolo,  # image
                                    str(classes[class_ids[i]]) + ', Confidence: ' + str(confidences[i]),    # text
                                    (x1, y1 - 10),                                                          # start position
                                    cv2.FONT_HERSHEY_SIMPLEX,                                               # font
                                    0.7,                                                                    # size
                                    colors[class_ids[i]],                                                   # BGR color
                                    1,                                                                      # thickness
                                    cv2.LINE_AA)                                                            # type of line
                SWIMMER_DETECTED = True                                                                     # Set True for YOLOv3

    #######################################################
    # Show Images
    #######################################################
    if conf["show_video"]:
        measured_FPS = (frames_processed / (time.time() - starting_time))   # Measure FPS that script runs at
        display_img = np.zeros((512, 512, 3), np.uint8)
        line1_text = "SWIMMER_DETECTED: {}".format(SWIMMER_DETECTED)        # Format text for screen putting
        line2_text = "NUMBER_SWIMMERS: {}".format(NUMBER_SWIMMERS)
        line3_text = "DROWNING_DETECT: {}".format(DROWNING_DETECT)
        line4_text = "FPS: {}".format(measured_FPS)
        line5_text = "Variance of Lap.: {}".format(var_of_laplacian)
        cv2.putText(display_img, line1_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, line2_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, line3_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, line4_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, line5_text, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Frames
        cv2.imshow("FF Motion Detection: Binary Image after Morphological Operations", thresh)
        cv2.imshow("Motion Detection: Absolute Difference", delta)
        cv2.imshow("BSMOG Mask", bs_mask)
        cv2.imshow("YOLOv3 Boxing", img_yolo)
        cv2.imshow("Color Detection: Binary image", binary_image)
        cv2.imshow("DDS: Underwater Video Feed", frame)
        cv2.imshow("Video + Environment Parameters", display_img)
        cv2.imshow("Logical AND'ing of Motion and Color Contours", binary_intersection)
        cv2.imshow("Averaging Motion Detection", thresh_avg)
        cv2.imshow("Color Detection: Image after Morphological Operations", binary_image)

    #######################################################
    # Write Files, Iterate on Frame Count
    #######################################################
    if conf["raspberry_pi"]:
        if frames_processed % 10 == 0:                           # Write To JSON File
            cv2.imwrite('../last_Image/last_Frame.jpg', frame)
            write_pool_info_json(SWIMMER_DETECTED, NUMBER_SWIMMERS, DROWNING_DETECT, SERIAL_NO, JSON_FILE_PATH)

    frames_processed += 1                                       # Iterate Frame Count

    action = cv2.waitKey(1)
    if action==27:
        break

cv2.destroyAllWindows()

