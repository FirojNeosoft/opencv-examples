import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('AB.jpg',0)

# Image Edge detection by canny
edges = cv2.Canny(img,100,200)

# Image Gradients
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

plt.subplot(231),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobelx gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobely gradient'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(sobelxy,cmap = 'gray')
plt.title('Sobelxy gradient'), plt.xticks([]), plt.yticks([])
plt.show()
