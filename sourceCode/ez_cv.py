# Underwater Swimmer Detection, Term Project
# Edge Detection Code for OptoSwim
#   Author: Alden Kane

import cv2 as cv2
import numpy as np
import json


def do_canny(frame):
    # Convert frame to grayscale for processing - Less Computationally Expensive
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 5x5 Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detector
    frame = cv2.Canny(blur, 50, 150)
    return frame


def segment_for_bottom(frame):
    # Take in a frame with canny edge detection applied, then apply mask to find horizontal lines at bottom of the pool
    # Take in a grayscale image
    height, width = frame.shape
    # Find all straight lines bottom 1/2 of Image
    bottom_rectangle = np.array([
        [(0, height), (width, height), (width, height/2), (0, height/2)]
                                ])
    mask = np.zeros_like(frame)
    # Fill polygon with 1s (white)
    mask = cv2.fillPoly(mask, np.int32([bottom_rectangle]), 255)
    frame = cv2.bitwise_and(frame, mask)
    # Return frame that now only has bottom area segmented for lines
    return frame


def find_bottom_line(frame):
    # Take in grayscale frame that is segmented for bottom, then use Hough Trasnform to find the best fit line for bottom
    height, width = frame.shape
    lines = cv2.HoughLines(frame, 1, np.pi / 180, 50)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame


def write_pool_info_json(swimDetected, numberSwimmers, drownDetected, filepath):
    # Function to format JSON data for transmission to Parse server
    # swimDetected = Bool
    # numberSwimmers = int
    # drownDetected = Bool
    # filepath = string
    with open(str(filepath), 'w', encoding='utf-8') as file:
        data = {
            'swimDetected':'{}'.format(swimDetected),
            'numberSwimmers':'{}'.format(numberSwimmers),
            'drownDetected':'{}'.format(drownDetected),
        }
        json.dump(data, file)

# def motion_detection(frame, firstFrame):
#     #Takes in color frame, converts to grayscale, applies motion detection
#     # Convert to grayscale, apply Gaussian Blur for processing. Saves computing power because motion is independent of color
#     # Gaussian smoothing will assist in filtering out high frequency noise from water moving, camera fluctuations, etc.
#     # TUNING: Can alter gaussian blur region for better detection --> Initially 21x21
#     gray_Img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray_Img = cv2.GaussianBlur(gray_Img, (15, 15), 0)
#     firstFrame = firstFrame
#
#     # Initialize first frame. This will only set the first frame once. Now, can compute difference between current frame and this.
#     if firstFrame is None:
#         firstFrame = gray_Img
#         continue
#
#     # Now comes absolute difference detection. This is "motion detection"
#     delta = cv2.absdiff(firstFrame, gray_Img)
#     thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)[1]