#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import cv2










path='/home/rick/桌面/tmp/yhy/t.jpeg'
path='/home/rick/桌面/tmp/yhy/t6.jpeg'
#path2='/home/rick/桌面/tmp/yhy/t2.jpeg'
path2='/home/rick/桌面/tmp/tk/t/image20160212004041.jpg'
#path2='/home/rick/桌面/tmp/tk/t/image20160212005701.jpg'

path2='/home/rick/桌面/tmp/tk/t/a.jpg'

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(path2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
_, binary2 = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY)
#binary2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1)

cv2.imshow("Image-New0", binary2)

Z = img2.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img2.shape))
gray2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

binary2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1)

cv2.imshow("Image-New1", binary2)


contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy2 = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_ = []
for contour in contours:
    if len(contour) > 100:
        contours_.append(contour)
contours2_ = []
for contour2 in contours2:
    if len(contour2) > 50 and len(contour2) < 200:
        contours2_.append(contour2)
print(len(contours_))
print(len(contours2_))


contour0 = contours_[1]
contour0_=[]
contours2__ = []
for contour2 in contours2_:
    similarity = cv2.matchShapes(contour0, contour2, cv2.CONTOURS_MATCH_I2, 0)
    if similarity <= 0.9:
        print('fffffffffff')
        contour0_ = contour2
    else:
        contours2__.append(contour2)
print(len(contour0))
print(len(contour0_))
cv2.drawContours(img, [contour0], -1, (0, 0, 255), 3)
try:
    cv2.drawContours(img2, [contour0_], -1, (0, 0, 255), 3)
except:
    print('No match!')
cv2.drawContours(img2, contours2__, -1, (0, 255, 0), 3)
cv2.imshow("Image-New", img)
cv2.imshow("Image-New2", img2)
cv2.waitKey()
cv2.destroyAllWindows()











#X = np.random.randint(25,50,(50,2))
#Y = np.random.randint(60,85,(50,2))
#Z = np.vstack((X,Y))
#
## convert to np.float32
#Z = np.float32(Z)
#
### define criteria and apply kmeans()
### 迭代次数为10次，精确度为1.0
##criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
##
### Now separate the data, Note the flatten()
##A = Z[label.ravel()==0]
##B = Z[label.ravel()==1]
##
### Plot the data
##plt.switch_backend('TkAgg')
##plt.scatter(A[:,0],A[:,1])
##plt.scatter(B[:,0],B[:,1],c = 'r')
##plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
##plt.xlabel('Height'),plt.ylabel('Weight')
##plt.show()
#
#
#
#
## 最大最小距离算法的Python实现
## 数据集形式data=[[],[],...,[]]
## 聚类结果形式result=[[[],[],...],[[],[],...],...]
## 其中[]为一个模式样本，[[],[],...]为一个聚类
#def start_cluster(data, t):
#    zs = [data[0]]  # 聚类中心集，选取第一个模式样本作为第一个聚类中心Z1
#    # 第2步：寻找Z2,并计算阈值T
#    T = step2(data, t, zs)
#    # 第3,4,5步，寻找所有的聚类中心
#    get_clusters(data, zs, T)
#    # 按最近邻分类
#    result = classify(data, zs, T)
#    return result
## 分类
#def classify(data, zs, T):
#    result = [[] for i in range(len(zs))]
#    for aData in data:
#        min_distance = T
#        index = 0
#        for i in range(len(zs)):
#            temp_distance = get_distance(aData, zs[i])
#            if temp_distance < min_distance:
#                min_distance = temp_distance
#                index = i
#        result[index].append(aData)
#    return result
## 寻找所有的聚类中心
#def get_clusters(data, zs, T):
#    max_min_distance = 0
#    index = 0
#    for i in range(len(data)):
#        min_distance = []
#        for j in range(len(zs)):
#            distance = get_distance(data[i], zs[j])
#            min_distance.append(distance)
#        min_dis = min(dis for dis in min_distance)
#        if min_dis > max_min_distance:
#            max_min_distance = min_dis
#            index = i
#    if max_min_distance > T:
#        zs.append(data[index])
#        # 迭代
#        get_clusters(data, zs, T)
## 寻找Z2,并计算阈值T
#def step2(data, t, zs):
#    distance = 0
#    index = 0
#    for i in range(len(data)):
#        temp_distance = get_distance(data[i], zs[0])
#        if temp_distance > distance:
#            distance = temp_distance
#            index = i
#    # 将Z2加入到聚类中心集中
#    zs.append(data[index])
#    # 计算阈值T
#    T = t * distance
#    return T
## 计算两个模式样本之间的欧式距离
#def get_distance(data1, data2):
#    distance = 0
#    for i in range(len(data1)):
#        distance += pow((data1[i]-data2[i]), 2)
#    return math.sqrt(distance)
#
#data = [[0, 0], [3, 8], [1, 1], [2, 2], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]]
#t = 0.5
#result = start_cluster(Z, t)
#plt.switch_backend('TkAgg')
#color = ['r','g','b']
#for i in range(len(result)):
##    print("----------第" + str(i+1) + "个聚类----------")
##    print(result[i])
#    r = result[i]
#    x = []
#    y = []
#    for t in r:
#        x.append(t[0])
#        y.append(t[1])
#    plt.scatter(np.array(x), np.array(y), c=color[i])
#plt.xlabel('Height')
#plt.ylabel('Weight')
#plt.show()