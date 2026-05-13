import cv2
import numpy as np

def segment(enhanced_img):
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask
