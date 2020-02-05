# Underwater Swimmer Detection, Term Project
# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# Author: Alden Kane
# This script detects motion on the assumption that the camera records from a still position. In actuality, this is how an underwater swimmer detection system would be constructed
# Referenced Adrian Rosebrock's "Basic motion detect and tracking with Python and OpenCV" tutorial at https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# Also adopted code snippets from "colorTracking.py" script by Dr. Adam Czajka, Andrey Kuelkahmp for University of Notre Dame's Fall 2019 CSE 40535/60535 course

import cv2
import numpy as np

#######################################################
# Pre-Processing: Video Capture, Declare Global Variables
#######################################################

# Alden's Video
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim1/swim1.1-16-of-24.mp4')
cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim2/swim2.1-3-of-5.mp4')

# Women's Underwater Hockey
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw1/uw1-16-of-84.mp4')
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw1/uw1-65-of-84.mp4')

# Men's Underwater Hockey
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw3/uw3-42-of-90.mp4')

# Initialize first frame - this is the basis of the still camera assumption for motion detection
firstFrame = None

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while (True):
    # Read image
    retval, img = cam.read()

    # rescale the input image if it's too large
    res_scale = 0.5
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    # Convert to grayscale, apply Gaussian Blur for processing. Saves computing power because motion is independent of color
    # Gaussian smoothing will assist in filtering out high frequency noise from water moving, camera fluctuations, etc.
    # TUNING: Can alter gaussian blur region for better detection --> Initially 21x21
    gray_Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_Img = cv2.GaussianBlur(gray_Img, (15, 15), 0)

    # Initialize first frame. This will only set the first frame once. Now, can compute difference between current frame and this.
    if firstFrame is None:
        firstFrame = gray_Img
        continue

    # Now comes absolute difference detection. This is "motion detection"
    delta = cv2.absdiff(firstFrame, gray_Img)
    thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)[1]

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

    # Perform some morphological operations
    # Remove noise from water
    thresh = cv2.erode(thresh, kernel_7, iterations=2)
    thresh = cv2.erode(thresh, kernel_9, iterations=1)

    # Close and dilate swimmer for better boxing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel_21)
    thresh = cv2.dilate(thresh, kernel_21, iterations=5) #Used to be 5
    thresh = cv2.dilate(thresh, kernel_40, iterations=4) #Used to be 4

    # Find contours
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Display Contours
    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    # Ignore bounding boxes smaller than "minObjectSize".
    # Tuned for optimal swimmer detection
    minObjectSize = 80

    #######################################################
    # Section 5: Object Detection
    #######################################################
    # If statement to detect if contours are present
    if contours:
        # Detect all swimmers, i.e. all objects with contours
        for contours in contours:

            # use just the first contour to draw a rectangle
            x, y, w, h = cv2.boundingRect(contours)

            # If statement to filter out small objects
            if w > minObjectSize or h > minObjectSize:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img,  # image
                            "Swimmer(s)",  # text
                            (x, y - 10),  # start position
                            cv2.FONT_HERSHEY_SIMPLEX,  # font
                            0.7,  # size
                            (0, 255, 0),  # BGR color
                            1,  # thickness
                            cv2.LINE_AA)  # type of line

    cv2.imshow("Underwater Video Feed", img)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Delta", delta)
    cv2.imshow("First Frame", firstFrame)

    action = cv2.waitKey(1)
    if action==27:
        break
