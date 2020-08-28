"""
    Lane Line Detection using Computer Vision
"""

import cv2
import numpy as np
import os
import sys

def ProcessFrame(img):
    # Copying image
    img_orig = img.copy()

    # Working on only the important parts
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)
    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.5]
    right_top = [cols * 0.6, rows * 0.6]
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    img = cv2.bitwise_and(img, mask)

    # Converting to HLS to make road colors stand out
    img_hls  = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Converting to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    table = np.array([((i / 255.0) ** 2.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_dark  = cv2.LUT(img, table)

    # Masking yellow and white colors (as they're likely to be road markings)
    lower_w = np.array([0, 200, 0])
    upper_w = np.array([200, 255, 255])
    lower_y = np.array([10, 0, 100])
    upper_y = np.array([40, 255, 255])
    mask_w = cv2.inRange(img_hls, lower_w, upper_w)
    mask_y = cv2.inRange(img_hls, lower_y, upper_y)
    mask = cv2.bitwise_or(mask_w, mask_y)
    img_masked = cv2.bitwise_and(img_dark, img_dark, mask=mask)

    # Performing Gassuian Blur
    img_blurred = cv2.GaussianBlur(img_masked, (7, 7), 0)

    # Detecting Edges
    img_canny = cv2.Canny(img_blurred, 70, 140)

    # Finding Straight Lines
    hough_lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, 20, np.array([]), 
                                minLineLength=20, maxLineGap=300)

    # Drawing lines on original image
    for line in hough_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_orig, (x1, y1), (x2, y2), [255, 0, 0], 2)
    return img_orig


if __name__ == "__main__":
    if len(sys.argv)<3:print("Usage:  ./main.py [i/v] (Image/Video) [source file]")
    else:
        if sys.argv[1] == 'i':
            cv2.imshow("Lane Detection", ProcessFrame(cv2.imread(sys.argv[2])))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cam = cv2.VideoCapture(sys.argv[2])
            while ret:
                _, frame = cam.read()
                cv2.imshow("Lane Detection", ProcessFrame(frame))
                cv2.waitKey(1)
            cv2.destroyAllWindows()
