import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('AB.jpg',0)
rows,cols = img.shape

# Use this for translation of an image
# M = np.float32([[1,0,400],[0,1,200]])
# dst = cv2.warpAffine(img,M,(cols,rows))

# Use this for rotation of an image
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

# Use this to resize of an image
# res = cv2.resize(img,(2*cols, 2*rows), interpolation = cv2.INTER_CUBIC)
# M = np.float32([[1,0,0],[0,1,0]])
# dst = cv2.warpAffine(res,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

