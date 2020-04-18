import cv2 as cv2
import numpy as np
from ez_cv import do_canny, segment_for_bottom, find_bottom_line

frame = cv2.imread("../annotated_100/images/goPOOLS1_2.jpg")
canny = do_canny(frame)
segment = segment_for_bottom(canny)

lines = cv2.HoughLinesP(segment, 1, np.pi / 180, threshold=10, minLineLength=100,maxLineGap=50)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(segment, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Debug", segment)
cv2.waitKey(0)
