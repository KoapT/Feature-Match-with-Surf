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


class Match(object):
    def __init__(self):
        super(Match, self).__init__()

    def _surf(self, img_arr):
        sift = cv2.xfeatures2d_SURF.create()
        keyPoint, descriptor = sift.detectAndCompute(img_arr, None)  # 特征提取得到关键点以及对应的特征向量
        return keyPoint, descriptor

    def save(self, img_arr, save_name):
        cv2.imwrite(save_name + '.jpg', img_arr)
        kp, des = self._surf(img_arr)
        pt = np.float32([m.pt for m in kp]).reshape(-1, 1, 2)
        np.savez(save_name, pt=pt, des=des)

    def load(self, mat_file):
        arr = np.load(mat_file + '.npz')
        pt, des = arr['pt'], arr['des']
        kps = []
        for i in pt:
            kp = cv2.KeyPoint()
            kp.pt = tuple(i[0])
            kps.append(kp)
        return pt, des, kps

    def _drawkeypoints(self, img_arr, kp):
        img = cv2.drawKeypoints(img_arr, kp, None)
        cv2.imshow('keypoints', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

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

    def _drawbias(self, mat_file, img_arr, chosen_point, mean_of_bias, isshow=False):
        img1 = cv2.imread(mat_file + '.jpg')
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
            cv2.imshow("pointmatche", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def calc_bias(self, mat_file, img_arr, drawbias=True, drawmatch=False):
        pt1, des1, kp1 = self.load(mat_file)
        kp2, des2 = self._surf(img_arr)
        pt2 = np.float32([m.pt for m in kp2]).reshape(-1, 1, 2)
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
        mask_max = bias < mean_of_bias + std_of_bias
        mask_min = bias > mean_of_bias - std_of_bias
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
            self._drawbias(mat_file, img_arr, chosen_point, mean_of_bias)

        # print(chosen_point)
        # print(mean_of_bias)
        # print(chosen_point + mean_of_bias)
        # print(bias)
        return mean_of_bias.tolist()


if __name__ == '__main__':
    img_b = cv2.imread('b.jpg')
    m = Match()
    # m.save(img_b, 'a')
    # pt, des, kp = m.load('b')
    # m.drawkeypoints(img_b, kp)
    t0 = time.time()
    bias = m.calc_bias('a', img_b, drawbias=True)
    print(bias)
    print(
        '''说明：
        bias的第一个数表示水平方向上的像素偏移，向右偏移为正；
        第二个数表示竖直方向上的像素偏移，向下偏移为正。'''
    )
    print('time cost: {:.2f}ms'.format(1000 * (time.time() - t0)))
