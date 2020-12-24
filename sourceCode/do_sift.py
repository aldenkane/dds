# do_sift.py
# Opto Labs Script
# Imports OpenCV, Initiates WebCam Read, and outputs to video stream
#   Author: Alden Kane

import cv2
import numpy as np

########################################
# Section 1: Open WebCam, Process and Display Video Feed in Loop
##########################################

# Video Capture
cam = cv2.VideoCapture(0)

while True:

    retval, frame = cam.read()
    cv2.imshow("Webcam Feed", frame)

    # cv2 Syntax for Processing
    action = cv2.waitKey(1)
    if action == 27:
        break

cv2.destroyAllWindows()
