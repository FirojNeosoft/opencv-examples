import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('blur_image.jpg',0)


# histogram equalization and use it to improve the contrast of our images
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('contrast_img.png',res)


# Contrast Limited Adaptive Histogram Equalization(CLAHE).It is advance technique to improve contrast of an image.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imwrite('clahe_2.jpg',cl1)
