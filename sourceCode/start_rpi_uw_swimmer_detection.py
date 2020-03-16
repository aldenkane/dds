# Underwater Swimmer Detection, Term Project
# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# Combination of detect_UW.py (Color Based Detection) and motionDetect_UW.py (Motion Based Swimmer Detection)
# Color-based detection code adapted from "colorTracking.py" script by Dr. Adam Czajka, Andrey Kuelkahmp for University of Notre Dame's Fall 2019 CSE 40535/60535 course
# Motion-based detection code referenced Adrian Rosebrock's "Basic motion detect and tracking with Python and OpenCV" tutorial at https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# Improvements made for OptoSwim
#   Author: Alden Kane

import cv2
import numpy as np
import math
import time
import logging

#######################################################
# Section 0.1: Logging + Camera Warmup
#######################################################

# Initiate Log Filename based on Time, Location
time_Tuple = time.localtime()
# Month.Date.Year_Hour.Sec.Min written in Logs Dir
log_Filename = '../logs/' + str(time_Tuple[1]) + '.' + str(time_Tuple[2]) + '.' + str(time_Tuple[0]) + '_' + str(time_Tuple[3]) + '.' + str(time_Tuple[4]) + '.' + str(time_Tuple[5]) + '_eye_V0.1.log'
# Configure Log
logging.basicConfig(filename=str(log_Filename), level=logging.DEBUG)
logging.debug('Accessed Log File')
# Allow for System Startup, Camera Warmup
time.sleep(10)

########################################
# Section 0.2: Webcam Capture
########################################
cam = cv2.VideoCapture(0)
logging.info('Accessed Camera')

#######################################################
# Section 0.3: Global Variables for Accuracy, Underwater Timing Features
#######################################################
# Motion Detection: Initialize first frame - this is the basis of the still camera assumption for motion detection
firstFrame = None
# Counter for frame iterator
frames_Processed = 0
# Global Variables for Timing Feature, Number of Swimmers in Pool
T = 0.00
drowningRisk = 0
FPS = 30
numSwimmers = 0
debounceTimer = 0

#######################################################
# Section 1: While Loop for Continuous Processing of Video Stream
#######################################################
while (True):
    #######################################################
    # Section 1.1: Color + Motion Detection - Read Image
    #######################################################
    # Set Starting Timer
    starting_Time = time.time()
    # Read image
    retval, img = cam.read()
    # Rescale Input Image
    res_scale = 0.5
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

    #######################################################
    # Section 1.2: Color Detection - Set up HSV Color Detection Bounds
    #######################################################
    # Declare hsv upper and lower bounds for color image detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([84, 0, 0])
    upper = np.array([111, 190, 150])
    binary_image = cv2.inRange(hsv, lower, upper)

    #######################################################
    # Section 1.3: Motion Detection - Grayscale Image Processing, Absolute Differencing for Motion Detection, Thresholding
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
    # Section 1.4: Color Tracking - Clean Up image with morphological operations
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

    #######################################################
    # Section 1.5: Motion Detection - Perform Morphological Operations, Find Contours, Draw Contours
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
    # Section 1.6: Perform Logical AND'ing of Binary Image, Implement Size Based Object Detection
    #######################################################
    # Perform bitwise AND of motion AND color contours
    binary_intersection = cv2.bitwise_and(thresh, binary_image)

    # Find contours
    contours, hierarchy = cv2.findContours(binary_intersection,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Ignore bounding boxes smaller than "minObjectSize". Tuned for optimal swimmer detection
    minObjectSize = 80

    #######################################################
    # Section 1.7: Object Detection and Localization w/ Drowning Detection Feature Built In
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
                # Put up bounding boxes w/ Text, If Statement for Timing
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
    # Section 1.8: Write 10th Frame to .jpg, Frame Counter
    #######################################################
    if frames_Processed%10 == 0:
        cv2.imwrite('../last_Image/last_Frame.jpg', img)
        logging.info('Wrote the ' + str(frames_Processed) + 'th frame')

    action = cv2.waitKey(1)
    if action==27:
        break

    # Global Counter of Frames Processed
    frames_Processed = frames_Processed + 1

cv2.destroyAllWindows()
