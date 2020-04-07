#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : surf_match.py
#   Author      : Koap
#   Created date: 2020/3/4 上午11:00
#   Description :
#
# ================================================================

import cv2
import numpy as np
import time
import os
import random

NUM_MATCH = 50
SOBEL_THRESH = 48
HOUGH_THRESH = 200    # "No line found"时，将 sobel_thresh和hough_thresh调低，可以得到更多直线。
VERTICAL_THRESH = 70  # 'No vertical line found!'时， 将vertical_thresh调高，可以降低对”垂直“的标准以得到更多直线。

class Match(object):
    def __init__(self):
        super(Match, self).__init__()

    def _surf(self, img_arr):
        sift = cv2.xfeatures2d_SURF.create()
        keyPoint, descriptor = sift.detectAndCompute(img_arr, None)  # 特征提取得到关键点以及对应的特征向量
        return keyPoint, descriptor
    
    def _houghline(self, img_arr):
        img_f32 = img_arr.astype(np.float32)
        b,g,r = cv2.split(img_f32)
        _, z = cv2.threshold(0.615*r-0.515*g-0.1*b, 0, 255, 3)
        z = z.astype(np.uint8)

        sobelx = cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = sobelx**2
        ret, sobelx = cv2.threshold(sobelx, SOBEL_THRESH, 255, 0)
        sobelx = sobelx.astype(np.uint8)

        lines = cv2.HoughLines(sobelx,1,np.pi/180,HOUGH_THRESH)
        if lines is None:
            print('No Line found!')
            return None

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
            if -VERTICAL_THRESH<=x2-x1<=VERTICAL_THRESH:
                n += 1
                X_sum += (x1+x2)/2
                X = X_sum/n 
                # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        if n==0:
            print('No vertical line found!')
            X = None
        return X

    def save(self, img_arr, save_name, line=False):
        kp, des = self._surf(img_arr)
        pt = np.float32([m.pt for m in kp]).reshape(-1, 1, 2)
        if line:
            line_x = self._houghline(img_arr)
        else:
            line_x = None
        if line_x is not None:
            self._drawline(img_arr,line_x)
        cv2.imwrite(save_name + '.jpg', img_arr)
        np.savez(save_name, pt=pt, des=des, line=line_x)

    def load(self, mat_file):
        arr = np.load(mat_file + '.npz', allow_pickle=True)
        pt, des ,line_x = arr['pt'], arr['des'], arr['line']
        kps = []
        for i in pt:
            kp = cv2.KeyPoint()
            kp.pt = tuple(i[0])
            kps.append(kp)
        return pt, des, kps, line_x

    def _drawkeypoints(self, img_arr, kp):
        img = cv2.drawKeypoints(img_arr, kp, None)
        cv2.imshow('keypoints', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def _drawline(self, img_arr, line_x):
        cv2.line(img_arr,(line_x,1000),(line_x,-1000),(0,255,0),1)

    def _drawmatch(self, mat_file, kp1, img2, kp2, good, matchesMask, isshow=False):
        img1 = cv2.imread(mat_file + '.jpg')
        draw_params = dict(
            matchesMask=matchesMask,
            flags=2)  # draw only inliers
        img = cv2.drawMatches(img1, kp1, img2, kp2, good, img2, **draw_params)
        cv2.imwrite(mat_file + '_match%d.jpg' % time.time(), img)
        if isshow:
            cv2.imshow("matches", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def _drawbias(self, mat_file, img_arr, chosen_point, mean_of_bias, line_x2, isshow=True):
        img1 = cv2.imread(mat_file + '.jpg')
        if line_x2 is not None:
            self._drawline(img_arr, line_x2)
        color = [(255, 0, 0),(0, 255, 0),(0, 0, 255)]
        for idx in range(chosen_point.shape[0]):
            point1_xy = tuple([int(i) for i in chosen_point[idx, :].tolist()])
            img1 = cv2.circle(img1, point1_xy, 5, color[idx], 1)
            point2_xy = tuple([int(i) for i in (chosen_point[idx, :] + mean_of_bias).tolist()])
            img2 = cv2.circle(img_arr, point2_xy, 5, color[idx], 1)
        img_h = max(img1.shape[0], img_arr.shape[0])
        img_w = img1.shape[1] + img_arr.shape[1]
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img[:img1.shape[0], :img1.shape[1], :] = img1
        img[:img_arr.shape[0], img1.shape[1]:, :] = img2
        cv2.imwrite(mat_file + '_pointmatch%d.jpg' % time.time(), img)
        if isshow:
            cv2.imshow("pointmatch", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def calc_bias(self, mat_file, img_arr, drawbias=False, drawmatch=False, calc_line_bias=False):
        pt1, des1, kp1, line_x1 = self.load(mat_file)
        kp2, des2 = self._surf(img_arr)
        pt2 = np.float32([m.pt for m in kp2]).reshape(-1, 1, 2)

        if calc_line_bias:
            line_x2 = self._houghline(img_arr)
        else:
            line_x2 = None

        # print(line_x2)
        print('1:',line_x1)
        print('2:',line_x2)
        if line_x2 == None or line_x1 == None:
            delta_line_x = None
        else:
            delta_line_x = line_x2- line_x1

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        ## knn match
        # knnMatches = bf.knnMatch(des1, des2, k = 1)
        # matches = [i[0] for i in knnMatches if i!=[]]
        good = matches[:NUM_MATCH]

        src_pts = pt1[[m.queryIdx for m in good], :, :]
        dst_pts = pt2[[m.trainIdx for m in good], :, :]

        # RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if drawmatch:
            self._drawmatch(mat_file, kp1, img_arr, kp2, good, matchesMask)

        bias = (dst_pts - src_pts)
        mask_bool = mask.astype(np.bool)
        bias = bias[mask_bool]
        src_pts = src_pts[mask_bool]

        # 过滤离群值
        mean_of_bias = np.mean(bias, axis=0)
        std_of_bias = np.std(bias, axis=0)
        mask_max = bias <= (mean_of_bias + std_of_bias)
        mask_min = bias >= (mean_of_bias - std_of_bias)
        mask_ = np.logical_and(
            np.logical_and(mask_max[:, 0], mask_max[:, 1]),
            np.logical_and(mask_min[:, 0], mask_min[:, 1])
        )
        bias = bias[mask_, :]
        src_pts = src_pts[mask_, :]

        chosen_n = random.randint(0, src_pts.shape[0] - 1)
        chosen_point = src_pts[chosen_n:chosen_n+3, :]

        mean_of_bias = np.mean(bias, axis=0)

        if drawbias:
            self._drawbias(mat_file, img_arr, chosen_point, mean_of_bias, line_x2)

        # print(chosen_point)
        # print(mean_of_bias)
        # print(chosen_point + mean_of_bias)
        # print(bias)
        return mean_of_bias.tolist(), delta_line_x


if __name__ == '__main__':
    img = cv2.imread('0318/img4.jpg')
    m = Match()
    # m.save(img, '0318/test4', line=True)
    # pt, des, kp = m.load('img')
    # m._drawkeypoints(img, kp)
    t0 = time.time()
    bias, delta_line_x = m.calc_bias('0318/img0', img, drawbias=True ,calc_line_bias=True)
    print('bias of pointmatch:',bias)
    print('bias of line match:',delta_line_x)
    print('time cost: {:.2f}ms'.format(1000 * (time.time() - t0)))
