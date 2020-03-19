#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

img = cv2.imread('/home/tk/projects/Feature-Match-with-Surf/20200317/img2.jpg')
b,g,r = cv2.split(img)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# sobelx = cv2.Sobel(r, cv2.CV_8U, 1, 0, ksize=3)
# ret, sobelx = cv2.threshold(sobelx, 15, 255, 0)
# canny = cv2.Canny(r, 50, 200)
sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)

sobelx = np.fabs(sobelx)
sobelx = sobelx.astype(np.uint8)
ret, sobelx = cv2.threshold(sobelx, 16, 255, 0)

lines = cv2.HoughLines(sobelx,1,np.pi/180,100)
lines = lines[:,0,:]

n=0
X_sum=0
for rho,theta in lines[:]: 
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    # print(x1,y1,x2,y2)
    if -10<=x2-x1<=10:
    	print(x1,y1,x2,y2)
        n += 1
        X_sum += (x1+x2)/2
        X = X_sum/n 
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
print('x coordinary:', X)
cv2.imshow('xx',sobelx)
cv2.imshow("pointmatche", img)
cv2.waitKey()
cv2.destroyAllWindows()
