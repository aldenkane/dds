import cv2
import numpy as np
import time

# Load YOLOv3 Retrained Model
net = cv2.dnn.readNet("yolov3_custom_train_final.weights", "yolov3_custom_train.cfg")

# Globals for resizing frames, counting frames
res_scale = 0.5
frames_processed = 0
# Sample once per second, i.e. every 30 frames
sample_rate = 30

classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print("Number of classes:",len(classes))
print(classes)

# Get the output layers of our YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading the camera
# For webcam!
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("/Users/aldenkane1/Documents/1optoSwim/dds/dataSet/swim3/swim3.1-12-of-14.mp4")
#cap = cv2.VideoCapture("/Volumes/Seagate HDD - Alden Kane/POOLS/maxPOOLS/Akaso Action Camera/akPools5.MOV")
#cap = cv2.VideoCapture('../dataSet/swim3/swim3.1-12-of-14.mp4')

# Font and random colors useful later when displaying the results
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    starting_time = time.time()

    # Read in Frames, Only Process Every 30th frame for 1 FPS
    retval, frame = cap.read()
    frames_processed = frames_processed + 1

    if frames_processed % sample_rate == 1:
        # Read and resize frame
        frame = cv2.resize(frame, (0, 0), fx=res_scale, fy=res_scale)
        height, width, channels = frame.shape

        '''
        *** Step 1: normalize input frame: use cv.dnn.blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]] to create a "blob" from image.
        Hints:
        -- scale the frame to something small, e.g. (220,220), for YOLOv3-tiny
        -- scale the frame to (320,320) for YOLOv3
        -- scale value of input images to the 0-1 range (from 0-255 range)
        -- we do not subtract mean from input images
        -- we need to swap R and B channels
        -- we don't need to crop after resize
        
        blob = ...
        
        '''

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(320,320), mean=0, swapRB=True, crop=False)


        '''
        *** Step 2: use cv.dnn_Net.setInput(blob[, name[, scalefactor[, mean]]]) to put your frame on the network's input.
        Hint:
        -- we no longer need to normalize the brightness
        
        '''

        net.setInput(blob)

        '''
        *** Step 3: use net.forward() to predict the outputs on the "output_layers"
        
        outputs = ...
        
        '''

        outputs = net.forward(output_layers)


        '''
        *** Step 4: The detection is done! The only thing to do is to display the results.
        
        Hints:
        
        Start with these lines:
        '''

        class_ids = []
        confidences = []
        boxes = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                '''
                Now for this particular object we have its class ID and confidence.
                But we want to ignore objects with low detection confidence. Thus, in the next lines
                process only those objects that have confidence higher than some threshold (e.g., 0.2).
        
                ...
        
                We may now calculate its box coordinates (x,y,w,h) multiplying values in "detection" vector by the frame width and height:
                '''
                if confidence >= 0.2:
                    #Multiply this by width and height
                    x = width*detection[0] #orresponds to X center
                    y = height*detection[1] #corresponds to Y center
                    w = width*detection[2] #corresponds to the box width, and
                    h = height*detection[3] #corresponds to the box height

                    '''
                    ...
                    Once ready, append information about this object to the list of all detected objects in this frame:
                    '''

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        '''
        
        
        
        '''
        '''
        *** Step 5: A single object can be represented by multiple boxes. We can remove this "noise"
        by using non-maximum suppression algorithm, implemented as:
        cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold[, eta[, top_k]]	)
        
        Hints:
        -- use the same "confThreshold" as above
        -- experiment with "nmsThreshold" (you can start with 0.4)
        
        indices = cv.dnn.NMSBoxes(...
        
        '''

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.2, eta=1, top_k=5)


        '''
        Step 6: Display boxes and class labels on the screen.
        
        Start with these lines:
        
        for i in range(len(boxes)):
            if i in indices:
        
            Hints:
            -- boxes' indices are in boxes[i]
            -- str(classes[class_ids[i]]) will give you the class label
            -- confidences[i] will give you the confidence score
            -- and you can use colors[class_ids[i]] for a box color
        
            Use "cv2.rectangle" and "cv2.text" to add the detection results to the "frame"
        
        '''

        for i in range(len(boxes)):
            if i in indices:
                for num_detects in indices:
                    x, y, w, h = boxes[i]
                    x1 = int(x-(w/2))
                    y1 = int(y-(h/2))
                    x2 = int(x+(w/2))
                    y2 = int(y+(h/2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_ids[i]], 3)
                    cv2.putText(frame,                                                                  # image
                                str(classes[class_ids[i]]) + ' -- Confidence: ' + str(confidences[i]),  # text
                                (x1, y1-10),                                                            # start position
                                cv2.FONT_HERSHEY_SIMPLEX,                                               # font
                                0.7,                                                                    # size
                                colors[class_ids[i]],                                                   # BGR color
                                1,                                                                      # thickness
                                cv2.LINE_AA)                                                            # type of line

        # We can also display the FPS:
        fps = 1 / (time.time() - starting_time)
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

        # Resize for own photos
        # scale_percent = 30  # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        frames_processed = frames_processed + 1
    else:
        frames_processed = frames_processed + 1

cap.release()
cv2.destroyAllWindows()
