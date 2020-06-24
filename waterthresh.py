#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/tk/Pictures/mutou.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# Otsu ' s 二值化；
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations = 1)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations = 1)
dist_transform = cv2.distanceTransform(opening,1,5)
ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg,sure_fg)
#Marker labeling
ret,makers1 = cv2.connectedComponents(sure_fg)
#Add one to all labels so that sure background is not 0 but 1;
markers = makers1 +1
#Now mark the region of unknow with zero;
markers[unknow ==255] =0
markers3 = cv2.watershed(img,markers)
img[markers3 == -1] =[255,0,0]
plt.subplot(1,3,1),
plt.imshow(makers1),
plt.title('makers1')
plt.subplot(1,3,2),
plt.imshow(markers3),
plt.title('markers3')
plt.subplot(1,3,3),
plt.imshow(img),
plt.title('img1'),
plt.show()