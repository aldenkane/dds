# Underwater Swimmer Detection, Term Project
# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# Combination of detect_UW.py (Color Based Detection) and motionDetect_UW.py (Motion Based Swimmer Detection)
# Improvements made for OptoSwim
#   Author: Alden Kane

import cv2
import numpy as np
import math
import socket
import imagezmq
import time
#from random import random
import datetime
from decimal import *

#######################################################
# Section 0: References
#######################################################
# Color-based detection code adapted from "colorTracking.py" script by Dr. Adam Czajka, Andrey Kuelkahmp for University of Notre Dame's Fall 2019 CSE 40535/60535 course
# Motion-based detection code referenced Adrian Rosebrock's "Basic motion detect and tracking with Python and OpenCV" tutorial at https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

#######################################################
# Global ZMQ Sending
#######################################################
# use either of the formats below to specifiy address of display computer
# sender = imagezmq.ImageSender(connect_to='tcp://jeff-macbook:5555')
# Alden's MacBook
sender = imagezmq.ImageSender(connect_to='tcp://10.10.65.247:5555')
rpi_name = socket.gethostname()  # send RPi hostname with each image

#######################################################
# Pre-Processing: Declare Many Video Captures (Uncomment Video to See), Global Variables
#######################################################
# Declare some underwater video captures for analysis

# Solo Swimmer, Good Response
# cam = cv2.VideoCapture('../dataSet/swim2/swim2.1-3-of-5.mp4')

# Two Swimmers, Begin Apart, Come Together at End, Fairly Good Response, Get Some Oscillation in Boxing
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.1-7-of-14.mp4')

# Two Swimmers, Begin Apart, Come Together, Then Separate, Very Good Response
cam = cv2.VideoCapture('../dataSet/swim3/swim3.1-12-of-14.mp4')

# Two Swimmers, Begin Apart, Come Together, Then Separate, Poor Response for Multiple Swimmers, Good Response for Solo
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.2-5-of-29.mp4')

# Solo Swimmer, Poor Response from Motion Detection in Middle
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.2-12-of-29.mp4')

# Solo Swimmer, Poor Response from Motion Detection in Middle
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.2-13-of-29.mp4')

# Two Solo Swimmers at Separate Times, Poor Response from 1st Swimmer, Good Response from Second Swimmer
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.3-2-of-3.mp4')

# Two Swimmers, Alright Response, As Male swimmer moves out of foreground, detection is lost
# cam = cv2.VideoCapture('../dataSet/swim3/swim3.4-3-of-14.mp4')

########################################
# Accuracy Videos at Rock
########################################
# cam = cv2.VideoCapture('../dataSet/swim4/swim4.3-5-of-9-30fps.mp4')
# cam = cv2.VideoCapture('../dataSet/swim4/swim4.4-2-of-4-30fps.mp4')
# cam = cv2.VideoCapture('../dataSet/swim4/swim4.5-7-of-10-30fps.mp4')
# cam = cv2.VideoCapture('../dataSet/swim4/swim4.5-4-of-10-30fps.mp4')

########################################
# Webcam Capture
########################################
# cam = cv2.VideoCapture(0)

# Motion Detection: Initialize first frame - this is the basis of the still camera assumption for motion detection
firstFrame = None

# Initialize Windows
cv2.namedWindow("Color Detection: Binary image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Detection: Binary image", 400, 225)

cv2.namedWindow("Color Detection: Image after Morphological Operations", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Detection: Image after Morphological Operations", 400, 225)

cv2.namedWindow("Motion Detection: Binary Image after Morphological Operations", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Detection: Binary Image after Morphological Operations", 400, 225)

cv2.namedWindow("Motion Detection: Absolute Difference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Detection: Absolute Difference", 400, 225)

cv2.namedWindow("Motion Detection: First Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Motion Detection: First Frame", 400, 225)

cv2.namedWindow("Logical AND'ing of Motion and Color Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Logical AND'ing of Motion and Color Contours", 400, 225)

cv2.namedWindow("DDS: Underwater Video Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("DDS: Underwater Video Feed", 400, 225)
cv2.moveWindow("DDS: Underwater Video Feed", 0, 0)


#######################################################
# Global Variables for Accuracy, Underwater Timing Features
#######################################################
# Counter for number of desired samples -- Global Used for accuracy calc.
N = 1

# Global Variables for Timing Feature, Number of Swimmers in Pool
T = 0.00
drowningRisk = 0
FPS = 30
numSwimmers = 0
debounceTimer = 0

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while (True):
    #######################################################
    # Pre-Processing - Initialize Window Position & Set Timers
    #######################################################
    starting_Time = time.time()

    cv2.moveWindow("Color Detection: Binary image", 840, 0)
    cv2.moveWindow("Color Detection: Image after Morphological Operations", 0, 300)
    cv2.moveWindow("Motion Detection: Binary Image after Morphological Operations", 420, 600)
    cv2.moveWindow("Motion Detection: Absolute Difference", 840, 300)
    cv2.moveWindow("Motion Detection: First Frame", 420, 300)
    cv2.moveWindow("Logical AND'ing of Motion and Color Contours", 420, 0)

    #######################################################
    # Section 1: Color + Motion Detection - Read Image
    #######################################################
    # Read image
    retval, img = cam.read()

    # Rescale Input Image
    res_scale = 0.5
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    #######################################################
    # Section 2: Color Detection - Set up HSV Color Detection Bounds
    #######################################################
    # Declare hsv upper and lower bounds for color image detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([84, 0, 0])
    upper = np.array([111, 190, 150])
    binary_image = cv2.inRange(hsv, lower, upper)

    #######################################################
    # Section 3: Motion Detection - Grayscale Image Processing, Absolute Differencing for Motion Detection, Thresholding
    #######################################################
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

    #######################################################
    # Section 4: Color Tracking - Show Binary Feed
    #######################################################
    # Debug: Show binary image video feed
    cv2.imshow("Color Detection: Binary image", binary_image)

    #######################################################
    # Section 5: Color Tracking - Clean Up image with morphological operations
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
    kernel_40 = np.ones((40, 40), np.uint8)
    kernel_71 = np.ones((71, 71), np.uint8)

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

    # Show binary image after morphological operations for debug
    cv2.imshow("Color Detection: Image after Morphological Operations", binary_image)

    #######################################################
    # Section 6: Motion Detection - Perform Morphological Operations, Find Contours, Draw Contours
    #######################################################
    # Perform some morphological operations
    # Remove noise from water
    thresh = cv2.erode(thresh, kernel_7, iterations=2)
    thresh = cv2.erode(thresh, kernel_11, iterations=1)

    # Close and dilate swimmer for better boxing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel_21)
    thresh = cv2.dilate(thresh, kernel_21, iterations=5)
    thresh = cv2.dilate(thresh, kernel_40, iterations=4)

    #######################################################
    # Section 7: Perform Logical AND'ing of Binary Image, Implement Size Based Object Detection
    #######################################################
    # Perform bitwise AND of motion AND color contours
    binary_intersection = cv2.bitwise_and(thresh, binary_image)

    # Find contours
    contours, hierarchy = cv2.findContours(binary_intersection,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Imshow
    cv2.imshow("Logical AND'ing of Motion and Color Contours", binary_intersection)

    # Ignore bounding boxes smaller than "minObjectSize". Tuned for optimal swimmer detection
    minObjectSize = 80

    #######################################################
    # Section 8: Motion Detection - Show Relevant Images
    #######################################################
    cv2.imshow("Motion Detection: Binary Image after Morphological Operations", thresh)
    cv2.imshow("Motion Detection: Absolute Difference", delta)
    cv2.imshow("Motion Detection: First Frame", firstFrame)

    #######################################################
    # Section 9: Object Detection and Localization w/ Drowning Detection Feature Built In
    #######################################################
    # If statement to detect if contours are present
    if contours:
        # Detect all swimmers, i.e. all objects with contours
        for contours in contours:

            # use just the first contour to draw a rectangle
            x, y, w, h = cv2.boundingRect(contours)

            # If statement to filter out small objects
            if w > minObjectSize or h > minObjectSize:
                T = T + 1                                   # Iterate on My Time
                scaled_T = math.ceil(T/FPS)                 # Scaled Time that Accounts for FPS

                if scaled_T >= 10:
                    drowningRisk = 1
                    drowningBox = (0,0,255)

                # Put up boudning boxes w/ Text, If Statement for Timing
                if not drowningRisk:
                    debounceTimer = (debounceTimer + 1) / FPS
                    if debounceTimer < 0.1:
                        T = T + 1
                        scaled_T = math.ceil(T / FPS)
                    elif debounceTimer > 1:
                        T = 0
                        drowningRisk = 0
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(img,  # image
                                "Swimmer(s)",  # text
                                (x, y - 10),  # start position
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # size
                                (0, 255, 0),  # BGR color
                                1,  # thickness
                                cv2.LINE_AA)  # type of line
                    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
                elif drowningRisk:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img,  # image
                                "Swimmer(s)",  # text
                                (x, y - 10),  # start position
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # size
                                (0, 0, 255),  # BGR color
                                1,  # thickness
                                cv2.LINE_AA)  # type of line
                    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

                # Get number of swimmers in pool from length of contours
                # numSwimmers = len(contours)

                # Measure FPS that Script Runs At:
                measured_FPS = (1 / (time.time() - starting_Time))

                # Text on Screen for Primitive Lifeguard UI
                line1_Text = "Time Underwater: {} second(s)".format(scaled_T)  # Format Text for Screen Putting
                line2_Text = "Drowning Risk: ({})".format(drowningRisk)
                line3_Text = "FPS: ({})".format(measured_FPS)
                cv2.putText(img, line1_Text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, line2_Text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, line3_Text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #######################################################
    # Section 10: Show Final DDS Underwater Video Feed, Resize and Move Windows for Display
    #######################################################
    cv2.imshow("DDS: Underwater Video Feed", img)

    #######################################################
    # Section 11: ImageZMQ Sending
    #######################################################
    sender.send_image(rpi_name, img)

    action = cv2.waitKey(1)
    if action==27:
        break

    action = cv2.waitKey(1)
    if action==27:
        break

    # #######################################################
    # # Section 11: Accuracy Metrics - Images are ClassifIed After, Logged in Excel
    # # UNCOMMENT TO RETURN 30 RANDOM FRAMES - ONLY NEEDED FOR ACCURACY CALCULATIONS
    # #######################################################
    #
    # # All videos are 30s long, w/ 30 FPS = 900 Frames/Video
    # # I want 30 frames per video to classify for Intersection over Union
    # # Below is code to give me 30 random frames
    #
    # # Generate random floating point number between 0 and 1
    # R = random()
    #
    # # Generate a threshold for the number of desired samples
    # nSamples = (FPS/900)
    #
    # # Declare a write location
    # write_location = "../accuracy/" + str(N) + ".jpg"
    #
    # # Only write 30 samples
    # if N <= 30:
    #     if R < nSamples:
    #         cv2.imwrite(str(write_location), img)
    #         N = N + 1

cv2.destroyAllWindows()