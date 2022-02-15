import cv2
import numpy as np
from matplotlib import pyplot as plt

#Input Gambarnya
img = cv2.imread('images.jpg',0)

#nilai hist dan bins
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

#After
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

img = cv2.imread('images.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) 
cv2.imshow('ujicoba.png',res)

plt.subplot(2,2,1)
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'center')

plt.subplot(2,2,2)
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

