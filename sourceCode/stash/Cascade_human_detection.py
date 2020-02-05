import cv2
import numpy as np
import time

# Old declaration w/ absolute path for person_cascade
# person_cascade = cv2.CascadeClassifier("C://Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/env/lib/python3.7/site-packages/cv2/data/haarcascade_fullbody.xml")
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Get video reading from file
cap = cv2.VideoCapture('/Users/aldenkane1/Documents/College/4SenSem1/Computer Vision/Project/dataSet/uw1/uw1-11-of-84.mp4')

# Get video reading from webcam
#cap = cv2.VideoCapture(0)

while True:
    r, frame = cap.read()
    if r:

        # Initialize Timer
        # start_time = time.time()

        frame = cv2.resize(frame,(640,360)) # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Haar-cascade classifier needs a grayscale image
        rects = person_cascade.detectMultiScale(gray_frame)

        # Timing - Meausres time that it took to process one frame of video (i.e. resize, put to gray, bound w/ rectangle)
        # end_time = time.time()
        # print("Elapsed Time:",end_time-start_time)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.imshow("Human Detection", frame)

    # Exit Conditions
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break
