# uw.py
# OptoSwim Embedded Module for Use in Pools
# Run off of Laptop w/ Screen
#   Author: Alden Kane

import cv2
import numpy as np
import math
import time
from ez_cv import do_canny, segment_for_bottom, find_bottom_line, generate_bottom_mask

########################################
# Section 1: Initialize Globals, Video Parameters, Windows
##########################################
cam = cv2.VideoCapture(0)   # Webcam Capture
first_frame = None          # Motion Detection First Frame
debounce_timer = 0          # Less Oscillation in Boxing
avg = None                  # Motion Detection Averaging Frame
frames_processed = 0        # Iterate on Frames Processed

# Global Variables for Timing Feature, Number of Swimmers in Pool
T = 0.00
drowningRisk = 0
FPS = 30
numSwimmers = 0


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

cv2.namedWindow("YOLOv3 Boxing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv3 Boxing", 400, 225)

cv2.namedWindow("Video + Environment Parameters", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video + Environment Parameters", 400, 225)

cv2.namedWindow("Edge Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Edge Detection", 400, 225)

#######################################################
# YOLO Initialization
#######################################################

# Load YOLOv3 Retrained Model
net = cv2.dnn.readNet("../yolo/yolov3_custom_train_final.weights", "../yolo/yolov3_custom_train.cfg")


res_scale = 0.5         # YOLO Processing Information

sample_rate = 30

classes = []
with open("../yolo/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print("Number of classes:",len(classes))
print(classes)

# Get the output layers of our YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Font and random colors useful later when displaying the results
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#######################################################
# While Loop for Continuous Processing of Video Stream
#######################################################
while True:
    #######################################################
    # Pre-Processing - Initialize Window Position & Set Timers
    #######################################################
    starting_time = time.time()

    cv2.moveWindow("Color Detection: Binary image", 840, 0)
    cv2.moveWindow("Color Detection: Image after Morphological Operations", 0, 300)
    cv2.moveWindow("Motion Detection: Absolute Difference", 840, 300)
    cv2.moveWindow("Motion Detection: First Frame", 420, 300)
    cv2.moveWindow("Logical AND'ing of Motion and Color Contours", 420, 0)
    cv2.moveWindow("Motion Detection: Binary Image after Morphological Operations", 0, 600)
    cv2.moveWindow("YOLOv3 Boxing", 420, 600)
    cv2.moveWindow("Video + Environment Parameters", 840, 600)
    cv2.moveWindow("Edge Detection", 1260, 0)
    cv2.moveWindow("DDS: Underwater Video Feed", 0, 0)

    #######################################################
    # Section 1: Color + Motion Detection - Read Image
    #######################################################
    # Read image
    retval, img = cam.read()
    r_yolo, img_yolo = cam.read()
    # Image for putting video parameters for debug
    display_img = np.zeros((512,512,3), np.uint8)

    #######################################################
    # Section 2: Color Detection - Set up HSV Color Detection Bounds
    #######################################################
    # Declare hsv upper and lower bounds for color image detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([84, 0, 0])
    upper = np.array([111, 255, 255])
    binary_image = cv2.inRange(hsv, lower, upper)

    #######################################################
    # Section 3: Motion Detection - Grayscale Image Processing, Absolute Differencing for Motion Detection, Thresholding
    #######################################################
    # Convert to grayscale, apply Gaussian Blur for processing. Saves computing power because motion is independent of color
    # Gaussian smoothing will assist in filtering out high frequency noise from water moving, camera fluctuations, etc.
    # TUNING: Can alter gaussian blur region for better detection --> Initially 21x21
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # Initialize first frame. This will only set the first frame once. Now, can compute difference between current frame and this.
    if first_frame is None:
        first_frame = gray
        continue

    # Now comes absolute difference detection. This is "motion detection"
    delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(delta, 100, 255, cv2.THRESH_BINARY)[1] #Used to be 10 --> Hyped this up for better motion detection

    # Averaging Motion Detection
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

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
    thresh = cv2.erode(thresh, kernel_3, iterations=2)
    thresh = cv2.erode(thresh, kernel_7, iterations=1)

    # Close and dilate swimmer for better boxing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel_21)
    thresh = cv2.dilate(thresh, kernel_11, iterations=5)
    thresh = cv2.dilate(thresh, kernel_13, iterations=4)

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
    cv2.imshow("Motion Detection: First Frame", first_frame)

    #######################################################
    # Section : Canny Edge Detection w/ Imshow
    #######################################################
    canny_img = do_canny(img)
    bottom_seg = segment_for_bottom(canny_img)
    #seg_with_lines = find_bottom_line(bottom_seg)
    seg_with_lines = generate_bottom_mask(img)
    cv2.imshow("Edge Detection", canny_img)

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

                # Put up bounding boxes w/ Text, If Statement for Timing
                if not drowningRisk:
                    debounce_timer = (debounce_timer + 1) / FPS
                    if debounce_timer < 0.1:
                        T = T + 1
                        scaled_T = math.ceil(T / FPS)
                    elif debounce_timer > 1:
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
                measured_FPS = (1 / (time.time() - starting_time))

                # Text on Screen for Primitive Lifeguard UI
                line1_Text = "Time Underwater: {} second(s)".format(scaled_T)  # Format Text for Screen Putting
                line2_Text = "Drowning Risk: ({})".format(drowningRisk)
                line3_Text = "FPS: ({})".format(measured_FPS)
                cv2.putText(display_img, line1_Text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_img, line2_Text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_img, line3_Text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if frames_processed % sample_rate == 1:
            # Get img shape for CV2 Blob
            height, width, channels = img_yolo.shape
            # Normalize input frame using blobFromImage, SwapRB Codes, and Scale Value to 1/255
            blob = cv2.dnn.blobFromImage(img_yolo, scalefactor=1 / 255, size=(320, 320), mean=0, swapRB=True, crop=False)
            # Set input of the net
            net.setInput(blob)
            # Predict outputs using net.forward
            outputs = net.forward(output_layers)
            # Initialize lists for displaying results, now that detection is done
            class_ids = []
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

            # Display image
            cv2.imshow("YOLOv3 Boxing", img_yolo)

    #######################################################
    # Section 10: Show Final DDS Underwater Video Feed, Resize and Move Windows for Display
    #######################################################
    cv2.imshow("DDS: Underwater Video Feed", img)
    cv2.imshow("Video + Environment Parameters", display_img)

    #######################################################
    # Section 11: Write 10th Frame to .jpg
    #######################################################
    if frames_processed%10 == 0:
        cv2.imwrite('../last_Image/last_Frame.jpg', img)

    action = cv2.waitKey(1)
    if action==27:
        break

    frames_processed += 1

cv2.destroyAllWindows()
