# Underwater Swimmer Detection, Term Project
# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# Author: Alden Kane
# Code adopted from "colorTracking.py" script by Dr. Adam Czajka, Andrey Kuelkahmp for University of Notre Dame's Fall 2019 CSE 40535/60535 course

import cv2
import numpy as np

#######################################################
# Preprocessing: Declare Video Capture
#######################################################

# Alden's Video
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim1/swim1.1-16-of-24.mp4')
cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim2/swim2.1-3-of-5.mp4')
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim2/swim2.1-5-of-5.mp4')

# Two Swimmers
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/swim3/swim3.4-3-of-14.mp4')

# Women's Underwater Hockey
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw1/uw1-16-of-84.mp4')
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw1/uw1-65-of-84.mp4')

# Men's Underwater Hockey
#cam = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw3/uw3-42-of-90.mp4')

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while (True):
    # Read image
    retval, img = cam.read()

    # rescale the input image if it's too large
    res_scale = 0.5
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    #######################################################
    # Section 1: Set up HSV color detection bounds
    #######################################################

    # Declare hsv upper and lower bounds for color image detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([84, 0, 0]) #Optimal config: 84, 0, 0
    upper = np.array([111, 190, 150]) #Optimal config: 111, 200, 150
    binary_image = cv2.inRange(hsv, lower, upper)

    #######################################################
    # Section 2: Show Binary Feed
    #######################################################

    # Debug: Show binary image video feed
    cv2.imshow("Binary image", binary_image)

    #######################################################
    # Section 3: Clean Up image with morphological operations
    #######################################################

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

    # Morphological operations for cleaner image
    # Remove artifacts of smaller pool lines, movement of water with initial erosion
    binary_image = cv2.erode(binary_image, kernel_3)

    # Fill in binary image of swimmers with closing
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=kernel_21)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel=kernel_21)

    # Erosion to clear up final image
    binary_image = cv2.erode(binary_image, kernel_5)

    # Dilation to Connect Swim Suits
    binary_image = cv2.dilate(binary_image, kernel_21, iterations=2)

    # Show binary image after morphological operations for debug
    cv2.imshow("Image after morphological operations", binary_image)

    #######################################################
    # Section 4: Find connected components, contours, display contours
    #######################################################

    # Find connected components
    cc = cv2.connectedComponents(binary_image)
    ccimg = cc[1].astype(np.uint8)

    # Find contours (OpenCV 4.x version)
    contours, hierarchy = cv2.findContours(ccimg,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Display Contours
    cv2.drawContours(img, contours, -1, (255,0,0), 3)

    # Ignore bounding boxes smaller than "minObjectSize".
    # Tuned for optimal swimmer detection
    minObjectSize = 100

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
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
                cv2.putText(img,            # image
                "Swimmer(s)",               # text
                (x, y-10),                  # start position
                cv2.FONT_HERSHEY_SIMPLEX,   # font
                0.7,                        # size
                (0, 255, 0),                # BGR color
                1,                          # thickness
                cv2.LINE_AA)                # type of line

    #######################################################
    # Section 6: Show Video Feed
    #######################################################

    cv2.imshow("Underwater Video Feed", img)

    action = cv2.waitKey(1)
    if action==27:
        break
