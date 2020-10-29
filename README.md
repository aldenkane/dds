# Opto Labs Embedded - Lean Production

## Authors

Computer vision by Alden Kane. Server and streaming by James Graham.  See "Resources and References," as well as code comments, for pertinent citations.

## Directions for Running CV Program on Raspberry Pi 4

**Library Requirements, using Python 3.7.5 Release:**

* numpy==1.17.4
* opencv-contrib-python==4.1.2.30

**Steps for Running Program:**

1. Clone Repository  
2. Ensure all libraries are installed  
3. Set `cam = cv2.VideoCapture('')` location of video to test, or enter 0 for a webcam feed
4. `cd sourceCode`
5. `python3 uw.py`

## Configuring Scripts Runnning at Boot for CV, Server, and Automatic Git Pull for FOTA

1. Add contents of `/boot/config/cront_w_pull_and_reboot.txt` to `crontab -e` file on Raspberry Pi, for Pi user

## Test Samples

* A test video, at `../dataSet/swim3/swim3.1-12-of-14.mp4` has been provided for running the code

## Parse install on Rasberry Pi

Install npm modules in project or root directory:
```
npm install parse --save
npm install btoa --save
```

## Execute Write to OptoSwim back4app Image class

Usage:
```
//require function from module
const sendImage = require('./parsePi.js');

//call function with filePath as argument
sendImage(filePath);

```

## Resources and Reference

* Dr. Adam Czajka (Notre Dame Computer Vision Research Lab), for his keen insight and expertise  
* Dr. Adrian Rosebrock (PyImageSearch) for his useful CV tutorials  
