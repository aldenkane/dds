# OptoSwim
# Dataset Engineering Scripts to write the kth frame to a .jpg
# Script takes 4 Command Line Arguements
# Run in same folder as movie, write to a folder that houses a metric fuck ton (MFT) of .jpgs
# Author: Alden Kane
# MIT License

import cv2 as cv2
import sys

#######################################################
# References
#######################################################

#######################################################
# Section 1: Fetch Command Line Arguments, Declare Global Constants
#######################################################


argUsage = ['jpg_From_Video.py', 'movieName - str', 'Frame Rate (FPS) - int', 'K Number for Sampling Frequency - int']
argList = sys.argv
nArgs = len(sys.argv)
frameRate = int(argList[2])                             #FPS
k = float(argList[3])                                   #Sampling Frequency in Hz
sF = frameRate/k                                        #Samples per second
N = 0                                                   #Frame Counter
J = 0                                                   #Image Number
movieName = argList[1]                                  #Movie name
strippedMovieName = movieName[:-4]                      #Movie name without filetype for writing .jpgs
pathToDataSet = '/Volumes/Seagate HDD - Alden Kane/POOLS/fallPOOLS/GoPro Hero'
pathToMovie = str(pathToDataSet) + '/' + str(movieName) #Path to the Movie rn
jpgFolder = str(pathToDataSet) + '/' + '../../mftJPGS'  #Folder to write .jpgs to

#######################################################
# Section 2: Error Handling for Command Line Arguments
#######################################################

argTypes = [type(movieName), type(frameRate), type(k)]
properArgTypes = "[<class 'str'>, <class 'int'>, <class 'float'>]"

if str(argTypes) != properArgTypes:
    print('Please use proper argument types. See list:')
    print(argUsage)
    sys.exit()

#######################################################
# Section 3: Video Capture and Write the Kth Jpg
#######################################################

# Declare VideoCapture
cam = cv2.VideoCapture(str(pathToMovie))

while True:
    # Read image
    retval, img = cam.read()

    if N%sF == 0:
        # Declare a write location and write image
        photoName = str(strippedMovieName) + '_' + str(J) + ".jpg"
        write_location = str(jpgFolder) + '/' + str(photoName)
        cv2.imwrite(str(write_location), img)
        J = J + 1                                                      #Increment Image Counter
        N = N + 1                                                      #Increment Frame Counter
        print('Wrote ' + str(photoName) + ', the ' + str(N) + 'th frame in the movie')

    else:
        N = N + 1
