import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('logo.jpg')

# Image Blurring OR Image Smoothing
gaussian_blur = cv2.GaussianBlur(img,(5,5),0)
bilateral_blur = cv2.bilateralFilter(img,9,75,75)
median_blur = cv2.medianBlur(img,5)

kernel = np.ones((5,5),np.float32)/25
averaging_blur = cv2.filter2D(img,-1,kernel)

# Removed noise from Image OR Image Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(231),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(gaussian_blur)
plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(bilateral_blur)
plt.title('Bilateral Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(median_blur)
plt.title('Median Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(averaging_blur)
plt.title('Averaging'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(dst)
plt.title('Denoising Image'), plt.xticks([]), plt.yticks([])
plt.show()
