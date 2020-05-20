# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017

import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture('/Volumes/Seagate HDD - Alden Kane/POOLS/maxPOOLS/Teslong USB Endoscope/tePools3.mov')

def print_menu():
    print("Options menu:")
    print("esc - Quit program")
    print("s - Snapshot")

def calc_histograms(img):
    color = ['b','g','r']
    colornames = ['Blue channel','Green channel','Red channel']
    plt.figure()
    for i,col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    plt.legend(colornames)
    plt.title('RGB Histogram')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channels = ['Hue','Saturation','Value']
    plt.figure()
    for i,ch in enumerate(channels):
        hist = cv2.calcHist([hsv], [i], None, [256], [0,256])
        plt.plot(hist, color=color[i])
    plt.title('HSV Histogram')
    plt.legend(channels)
    plt.show()


if __name__ == '__main__':

    print_menu()

    while (True):
        retval, img = cam.read()
        imcrop = None

        res_scale = 0.6             # rescale the input image if it's too large
        img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

        cv2.imshow("Preview", img)

        action = cv2.waitKey(1)
        if action == 27:    # escape
            break
        elif action == ord('h'):    # help
            print_menu()
        elif action == ord('s'):    # snapshot
            still = img
            r = cv2.selectROI(still)
            imcrop = still[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            cv2.imshow("ROI",imcrop)

        if imcrop is not None:
            calc_histograms(imcrop)

    cv2.destroyAllWindows()
