# DDS: A Two-Feature Underwater Swimmer Detection Systems

CSE 40535 Term Project

## Authors

All source code by Alden Kane. See "Resources and References," as well as code comments, for pertinent citations.

[https://www.aldenkane.com](https://www.aldenkane.com)

## Sensor Selection and Data Collection

I used a GoPro Hero7 Black (1920 x 1080 at 30 or 60 FPS) for data collection. I collected approximately 1.5 hours of underwater video at Rolf's Acquatic Center and Rockne Memorial Pool at the University of Notre Dame.

## Directions on Running Program

_Library Requirements, using Python 3.7.5 Release:_
numpy==1.17.4
opencv-contrib-python==4.1.2.30

_Steps for Running Program:_
..1. Clone Repository
..2. Ensure all libraries are installed
..3. Set `cam = cv2.VideoCapture('')` location of video to test
..4. From the top-level directory:
```
python3 uw_swimmer_detection.py
```

_Test Samples_

..* A test video, at `../dataSet/swim3/swim3.1-12-of-14.mp4` has been provided for running the code

## System Accuracy

Testing by Alden Kane showed an Intersection over Union (IoU) of approximately 0.81 ± 0.12 on the train and validation sets. The test set, sourced from Rockne Memorial Pool, show lower classificaiton accuracy due to poor overhead illumination in the pool.

## Future Improvements

System improvements can be generated using:
..* A convolutional neural network (CNN) for swimmer classification. This entails annotating a data set large enough for feature, and then integrating it with color and motion detection in a majority-voting system.
..* A more diverse data set of swimmers with different skin pigment, age, and swimming style.
..* Improved sensor selection. Selecting embedded cameras with lower resolution and frame rates can enhance run time, save on computational cost, and allow for scaleable architectures for implementation in commercial pools.

## Resources and Reference

..* Dr. Adam Czajka (Notre Dame Computer Vision Research Lab), for his keen insight and expertise
..* Dr. Adrian Rosebrock (PyImageSearch) for his useful CV tutorials

