# Opto.py
# Python Library for OptoSwim's V0.1 Modules and Prototypes
# Property of OptoSwim
#   Author: Alden Kane

import cv2 as cv2
import numpy as np
import json

# Function to Run YOLO for dds
def dds_yolo(frame, net, output_layers, classes):
    height, width, channels = frame.shape   # Get img shape for CV2 Blob
    yolo_swimmer_count = 0
    # Normalize input frame using blobFromImage, SwapRB Codes, and Scale Value to 1/255
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(320, 320), mean=0, swapRB=True, crop=False)
    net.setInput(blob)                      # Set input of the net
    outputs = net.forward(output_layers)    # Predict outputs using net.forward
    class_ids = []                          # Initialize lists for displaying results, now that detection is done
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]          # Get scores of detection
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Now, have class ID's and have detections. Now, ignore, scores of low confidence. Get bounding box coordinates here
            if confidence >= 0.2:
                # Multiply this by width and height
                x = width * detection[0]    # Corresponds to X center
                y = height * detection[1]   # Corresponds to Y center
                w = width * detection[2]    # Corresponds to the box width
                h = height * detection[3]   # Corresponds to the box height

                # Append info to boxes list (List w/ in list)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Top_k controls how many boxes/swimmers can be detected
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.2, eta=1, top_k=10)

    # Perform Boxing and Display
    for i in range(len(boxes)):
        if i in indices:
            for num_detects in indices:
                x, y, w, h = boxes[i]
                x1 = int(x - (w / 2))
                y1 = int(y - (h / 2))
                x2 = int(x + (w / 2))
                y2 = int(y + (h / 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(frame,  # image
                            str(classes[class_ids[i]]) + ', Confidence: ' + str(confidences[i]),  # text
                            (x1, y1 - 10),              # start position
                            cv2.FONT_HERSHEY_SIMPLEX,   # font
                            0.7,                        # size
                            (255, 0, 0),                # BGR color
                            1,                          # thickness
                            cv2.LINE_AA)                # type of line
                yolo_swimmer_count += 1

    return frame, yolo_swimmer_count


def write_pool_info_json(swimDetected, numberSwimmers, drownDetected, serialNo, filepath):
    # Function to format JSON data for transmission to Parse server
    # swimDetected = Bool
    # numberSwimmers = int
    # drownDetected = Bool
    # filepath = string
    with open(str(filepath), 'w', encoding='utf-8') as file:
        data = {
            'swimDetected': bool(swimDetected),
            'numberSwimmers': '{}'.format(numberSwimmers),
            'drownDetected': bool(drownDetected),
            'serialNo': '{}'.format(serialNo)
        }
        json.dump(data, file)


