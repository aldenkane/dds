# eye.py
# Production code for Raspberry Pi OptoSwim Eye Module
#   Author: Alden Kane
#   Property of OptoSwim

import cv2
import numpy as np
import math
import time
import logging
from ez_cv import write_pool_info_json

#######################################################
# Coding Conventions
#######################################################
''' -All variables transported off of module are UPPER_CASE
    -Algorithms that take more than 10 lines are turned into functions
    -All functions housed in opto.py
    -frame used as default var name for CV processing
'''

#######################################################
# Initiate Logging + Camera Read + Declare Globals
#######################################################
time_tuple = time.localtime()                           # Unpack time tuple
log_filename = '../logs/' + str(time_tuple[1]) + '.' + str(time_tuple[2]) + '.' + str(time_tuple[0]) + '_' + str(time_tuple[3]) + '.' + str(time_tuple[4]) + '.' + str(time_tuple[5]) + '_eye_V0.1.log'
logging.basicConfig(filename=str(log_filename), level=logging.DEBUG)

time.sleep(10)                                          # Camera warmup
cam = cv2.VideoCapture(0)                               # Camera capture
logging.info('Accessed Camera')

frames_processed = 0                                    # Computer vision global variables
first_frame = None                                      # Motion detection: first frame for still frame assumption
T = 0.00                                                # For timing
debounce_timer = 0                                      # Later used for less oscillation in boxing
FPS = 30

SERIAL_NO = '0001'                                      # JSON writing globals
JSON_FILE_PATH = '../last_Image/event.json'
NUMBER_SWIMMERS = 0
SWIMMER_DETECTED = False
DROWNING_DETECT = False

kernel_3 = np.ones((3, 3), np.uint8)                    # Declare kernels
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_11 = np.ones((11, 11), np.uint8)
kernel_21 = np.ones((21, 21), np.uint8)
kernel_40 = np.ones((40, 40), np.uint8)

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while (True):
    starting_time = time.time()                          # Used for FPS calc
    retval, frame = cam.read()
    res_scale = 0.5                                      # Rescale Input Image
    frame = cv2.resize(frame, (0, 0), fx=res_scale, fy=res_scale)

    #######################################################
    # Color Detection - Set up HSV Color Detection Bounds
    #######################################################
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)          # Declare hsv upper and lower bounds for color image detection
    lower = np.array([84, 0, 0])
    upper = np.array([111, 190, 150])
    binary_image_color = cv2.inRange(hsv, lower, upper)

    #######################################################
    # Motion Detection - Grayscale Image Processing, Absolute Differencing for Motion Detection, Thresholding
    #######################################################
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Get grayscale frame
    gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)  # Gaussian blur to filter out high freq noise from water moving, camera fluctuations

    # Initialize first frame. This will only set the first frame once. Now, can compute difference between current frame and this
    if first_frame is None:
        first_frame = gray_frame
        continue

    # Now comes absolute difference detection. This is "motion detection"
    delta = cv2.absdiff(first_frame, gray_frame)
    binary_image_motion = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)[1]

    #######################################################
    # Color Tracking - Clean Up image with morphological operations
    #######################################################

    # Morphological operations for cleaner image
    binary_image_color = cv2.erode(binary_image_color, kernel_3)                                    # Remove artifacts of smaller pool lines, movement of water with initial erosion
    binary_image_color = cv2.morphologyEx(binary_image_color, cv2.MORPH_CLOSE, kernel=kernel_21)    # Fill in binary image of swimmers with closing
    binary_image_color = cv2.morphologyEx(binary_image_color, cv2.MORPH_CLOSE, kernel=kernel_21)

    # Erosion to clear up final image
    binary_image_color = cv2.erode(binary_image_color, kernel_5)

    # Dilation to Connect Swim Suits - This adds noise to this binary image, but this is filtered out by logical AND later
    binary_image_color = cv2.dilate(binary_image_color, kernel_21, iterations=2)

    #######################################################
    # Section 1.5: Motion Detection - Perform Morphological Operations, Find Contours, Draw Contours
    #######################################################
    binary_image_motion = cv2.erode(binary_image_motion, kernel_7, iterations=2)                        # Morphological operations for noise reduction
    binary_image_motion = cv2.erode(binary_image_motion, kernel_11, iterations=1)
    binary_image_motion = cv2.morphologyEx(binary_image_motion, cv2.MORPH_CLOSE, kernel=kernel_21)      # Close and dilate swimmer for better boxing
    binary_image_motion = cv2.dilate(binary_image_motion, kernel_21, iterations=5)
    binary_image_motion = cv2.dilate(binary_image_motion, kernel_40, iterations=4)

    #######################################################
    # Section 1.6: Perform Logical AND'ing of Binary Image, Implement Size Based Object Detection
    #######################################################
    # Perform bitwise AND of motion AND color contours
    binary_intersection = cv2.bitwise_and(binary_image_motion, binary_image_color)

    # Find contours
    contours, hierarchy = cv2.findContours(binary_intersection,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    minObjectSize = 80                                      # Ignore bounding boxes smaller than "minObjectSize"

    #######################################################
    # Section 1.7: Object Detection and Localization w/ Drowning Detection Feature Built In
    #######################################################
    # If statement to detect if contours are present
    if contours:
        # Detect all swimmers, i.e. all objects with contours
        NUMBER_SWIMMERS = 0
        for contours in contours:
            # use just the first contour to draw a rectangle
            x, y, w, h = cv2.boundingRect(contours)
            # If statement to filter out small objects
            if w > minObjectSize or h > minObjectSize:
                T = T + 1                                   # Iterate on My Time
                scaled_T = math.ceil(T/FPS)                 # Scaled Time that Accounts for FPS
                NUMBER_SWIMMERS = NUMBER_SWIMMERS + 1       # For each bounding box of sufficient size, iterate on swimmers
                if scaled_T >= 15:
                    DROWNING_DETECT = True
                    drowningBox = (0, 0, 255)
                # Put up bounding boxes w/ Text, If Statement for Timing
                # Logic to handle oscillating boxes
                if not DROWNING_DETECT:
                    debounceTimer = (debounceTimer + 1) / FPS
                    if debounceTimer < 0.1:
                        T = T + 1
                        scaled_T = math.ceil(T / FPS)
                    elif debounceTimer > 1:
                        T = 0
                        DROWNING_DETECT = False
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame,  # image
                                "Swimmer(s)",  # text
                                (x, y - 10),  # start position
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # size
                                (0, 255, 0),  # BGR color
                                1,  # thickness
                                cv2.LINE_AA)                # type of line
                    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)
                elif DROWNING_DETECT:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame,  # image
                                "Swimmer(s)",  # text
                                (x, y - 10),  # start position
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # size
                                (0, 0, 255),  # BGR color
                                1,  # thickness
                                cv2.LINE_AA)  # type of line
                    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

                # Measure FPS that Script Runs At:
                measured_FPS = (1 / (time.time() - starting_time))

                # Text on Screen for Primitive Lifeguard UI
                line1_Text = "Time Underwater: {} second(s)".format(scaled_T)  # Format Text for Screen Putting
                line2_Text = "Drowning Risk: ({})".format(DROWNING_DETECT)
                line3_Text = "FPS: ({})".format(measured_FPS)
                cv2.putText(frame, line1_Text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, line2_Text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, line3_Text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #######################################################
    # Section 1.8: Write 10th Frame to .jpg, Frame Counter
    #######################################################
    if frames_processed % 10 == 0:
        cv2.imwrite('../last_Image/last_Frame.jpg', frame)
        logging.info('Wrote the ' + str(frames_processed) + 'th frame')
        SWIMMER_DETECTED = bool(NUMBER_SWIMMERS)
        write_pool_info_json(SWIMMER_DETECTED, NUMBER_SWIMMERS, DROWNING_DETECT, SERIAL_NO, JSON_FILE_PATH)

    frames_processed += 1  # Iterate on frames processed

