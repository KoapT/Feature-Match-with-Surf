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

NUM_MATCH = -1  # -1表示所有点
SOBEL_THRESH = .2
HOUGH_THRESH = 179  # "No line found"时，将 sobel_thresh和hough_thresh调低，可以得到更多直线。
VERTICAL_THRESH = 70  # 'No vertical line found!'时， 将vertical_thresh调高，可以降低对”垂直“的标准以得到更多直线。
PROPORTION = [.05, .5]  # 检测矩形框区域相对整个图案的最小、最大占比
KERNEL_morphologyEx = [8, 8]  # 形态学变换的核
THETA = .7


def walk(dir_path):
    file_list = []
    for (root, dirs, files) in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


class Match(object):
    def __init__(self):
        super(Match, self).__init__()

    def _surf(self, img_arr):
        surf = cv2.xfeatures2d_SURF.create()
        keyPoint, descriptor = surf.detectAndCompute(img_arr, None)  # 特征提取得到关键点以及对应的特征向量
        return keyPoint, descriptor

    def _houghline(self, img_arr):
        img_f32 = img_arr.astype(np.float32)
        b, g, r = cv2.split(img_f32)
        _, z = cv2.threshold(0.615 * r - 0.515 * g - 0.1 * b, 0, 255, 3)
        z = z.astype(np.uint8)

        sobelx = cv2.Sobel(z, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.fabs(sobelx)
        _max = np.max(sobelx)
        _min = np.min(sobelx)
        sobelx = 1. * (sobelx - _min) / (_max - _min)
        ret, sobelx = cv2.threshold(sobelx, SOBEL_THRESH, 255, 0)
        sobelx = sobelx.astype(np.uint8)

        lines = cv2.HoughLines(sobelx, 1, np.pi / 180, HOUGH_THRESH)
        if lines is None:
            print('No Line found!')
            return None

        lines = lines[:, 0, :]

        n = 0
        X_sum = 0
        for rho, theta in lines[:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = x0 + 1000 * (-b)
            y1 = y0 + 1000 * (a)
            x2 = x0 - 1000 * (-b)
            y2 = y0 - 1000 * (a)
            # print(x1,y1,x2,y2)
            if -VERTICAL_THRESH <= x2 - x1 <= VERTICAL_THRESH:
                n += 1
                X_sum += (x1 + x2) / 2
        X = int(X_sum / n)
        if n == 0:
            print('No vertical line found!')
            X = None
        return X

    def _iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area + 1e-06

        return inter_area / union_area

    def findRectangle(self, img_arr):
        '''
        :return: a list
        '''
        rec_boxes = []
        H, W, _ = img_arr.shape
        img_area = H * W
        imgray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones(KERNEL_morphologyEx, np.uint8)
        img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        img_edge = cv2.Canny(img_opening, 40, 20)
        image, contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = box = cv2.boundingRect(contour)
            if img_area * PROPORTION[0] < w * h < img_area * PROPORTION[1] and .8 < 1.0 * w / h < 2.5:
                if box not in rec_boxes:
                    rec_boxes.append(box)
        return rec_boxes

    def _drawRectangle(self, img, rec_boxes):
        '''
        :param img: type->array
        :param rec_boxes:  type->list
        '''
        if not isinstance(rec_boxes, list):
            rec_boxes.tolist()
        if len(rec_boxes) == 0:
            return
        for rec in rec_boxes:
            x, y, w, h = rec
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def save(self, img_arr, save_name, line=False):
        assert img_arr.ndim == 3, "Expect a 3-channel image array,but something else input!!!"
        kp, des = self._surf(img_arr)
        pt = np.float32([m.pt for m in kp]).reshape(-1, 1, 2)
        rec_boxes = self.findRectangle(img_arr)
        if line:
            line_x = self._houghline(img_arr)
        else:
            line_x = None
        if line_x is not None:
            self._drawline(img_arr, line_x)
        self._drawRectangle(img_arr, rec_boxes)
        cv2.imwrite(save_name + '.jpg', img_arr)
        np.savez(save_name, pt=pt, des=des, line=line_x, rec=rec_boxes)

    def load(self, mat_file):
        arr = np.load(mat_file + '.npz', allow_pickle=True)
        pt, des, line_x, rec_boxes = arr['pt'], arr['des'], arr['line'], arr['rec']
        kps = []
        for i in pt:
            kp = cv2.KeyPoint()
            kp.pt = tuple(i[0])
            kps.append(kp)
        return pt, des, kps, line_x.item(), rec_boxes.tolist()

    def _drawkeypoints(self, img_arr, kp, isshow=False):
        img = cv2.drawKeypoints(img_arr, kp, None)
        if isshow:
            cv2.imshow('keypoints', img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def _drawline(self, img_arr, line_x):
        cv2.line(img_arr, (line_x, 1000), (line_x, -1000), (0, 255, 0), 1)

    def _drawmatch(self, mat_file, kp1, img2, kp2, matches, matchesMask, isshow=False):
        img1 = cv2.imread(mat_file + '.jpg')
        draw_params = dict(
            matchesMask=matchesMask,
            flags=2)  # draw only inliers
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, **draw_params)
        cv2.imwrite(mat_file + '_match%d.jpg' % (time.time()*100), img)
        if isshow:
            cv2.imshow("matches", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def _drawbias(self, mat_file, img_arr, chosen_point, mean_of_bias, rec_boxes2, line_x2, delta_line_x, isshow=False):
        img1 = cv2.imread(mat_file + '.jpg')
        img2 = img_arr
        if len(rec_boxes2) > 0:
            self._drawRectangle(img2, rec_boxes2)
        if line_x2 is not None:
            self._drawline(img2, line_x2)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (0, 255, 255), (255, 0, 255),
                 (0, 0, 0), (255, 255, 255), (255, 128, 128)]
        for idx in range(chosen_point.shape[0]):
            point1_xy = tuple([int(i) for i in chosen_point[idx, :].tolist()])
            img1 = cv2.circle(img1, point1_xy, 5, color[idx % 9], 2)
            point2_xy = tuple([int(i) for i in (chosen_point[idx, :] + mean_of_bias).tolist()])
            img2 = cv2.circle(img2, point2_xy, 5, color[idx % 9], 2)
        img_h = max(img1.shape[0], img_arr.shape[0])
        img_w = img1.shape[1] + img_arr.shape[1]
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img[:img1.shape[0], :img1.shape[1], :] = img1
        img[:img_arr.shape[0], img1.shape[1]:, :] = img2
        text1 = 'bias of pointmatch:{}'.format(mean_of_bias[0])
        # text2 = 'bias of linematch:{}'.format(delta_line_x)
        cv2.putText(img, text1, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        # cv2.putText(img, text2, (15,30),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        cv2.imwrite(mat_file + '_pointmatch%d.jpg' % (time.time()*100), img)
        if isshow:
            cv2.imshow("pointmatch", img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def calc_bias(self, mat_file, img_arr, drawbias=False, drawmatch=False, calc_line_bias=False):
        assert img_arr.ndim == 3, "Expect a 3-channel image array,but something else input!!!"
        pt1, des1, kp1, line_x1, rec_boxes1 = self.load(mat_file)
        kp2, des2 = self._surf(img_arr)
        pt2 = np.float32([m.pt for m in kp2]).reshape(-1, 1, 2)
        rec_boxes2 = self.findRectangle(img_arr)

        if calc_line_bias:
            line_x2 = self._houghline(img_arr)
        else:
            line_x2 = None
        if line_x2 is None or line_x1 is None:
            delta_line_x = None
        else:
            delta_line_x = line_x2 - line_x1

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        ## knn match
        # knnMatches = bf.knnMatch(des1, des2, k = 1)
        # matches = [i[0] for i in knnMatches if i!=[]]

        src_pts = pt1[[m.queryIdx for m in matches], :, :]
        dst_pts = pt2[[m.trainIdx for m in matches], :, :]

        # RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        if drawmatch:
            self._drawmatch(mat_file, kp1, img_arr, kp2, matches, matchesMask)

        bias = dst_pts - src_pts
        mask_bool = mask.astype(np.bool)
        bias = bias[mask_bool]
        src_pts = src_pts[mask_bool]
        dst_pts = dst_pts[mask_bool]
        if src_pts.shape[0] == 0:
            print('No matched point found!!!')
            return [None, None], delta_line_x

        if len(rec_boxes1) == 0 or len(rec_boxes2) == 0:
            print('No hot area found in the picture!!! The result could be Unreliable!!!')
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
            mean_of_bias = np.mean(bias, axis=0)

        else:
            truemasks = []
            for rec1 in rec_boxes1:
                x1, y1, w1, h1 = rec1
                recMask_x1 = np.logical_and(0 <= src_pts[:, 0] - x1, src_pts[:, 0] - x1 <= w1)
                recMask_y1 = np.logical_and(0 <= src_pts[:, 1] - y1, src_pts[:, 1] - y1 <= h1)
                recMask1 = np.logical_and(recMask_x1, recMask_y1)
                match_num = 0
                for rec2 in rec_boxes2:
                    x2, y2, w2, h2 = rec2
                    recMask_x2 = np.logical_and(0 <= dst_pts[:, 0] - x2, dst_pts[:, 0] - x2 <= w2)
                    recMask_y2 = np.logical_and(0 <= dst_pts[:, 1] - y2, dst_pts[:, 1] - y2 <= h2)
                    recMask2 = np.logical_and(recMask_x2, recMask_y2)
                    recMask = np.logical_and(recMask1, recMask2)
                    if len(recMask[recMask == True]) >= match_num:
                        match_num = len(recMask[recMask == True])
                        truemask = recMask
                truemasks.append(truemask)
            goodmask = np.max(np.array(truemasks), axis=0)  # 相当于逻辑或
            nogoodmask = np.logical_not(goodmask)
            bias_good = bias[goodmask]
            bias_nogood = bias[nogoodmask]
            src_pts_good = src_pts[goodmask]
            src_pts_nogood = src_pts[nogoodmask]
            n_good = bias_good.shape[0]
            n_nogood = bias_nogood.shape[0]
            if n_good == 0:
                print('No matched point found in the hot area!')
                mean_of_bias = np.mean(bias_nogood, axis=0)
            elif n_nogood == 0:
                print('No matched point found out of the hot area!')
                mean_of_bias = np.mean(bias_good, axis=0)
            else:
                mean_of_bias_good = np.mean(bias_good, axis=0)
                mean_of_bias_nogood = np.mean(bias_nogood, axis=0)
                delta = mean_of_bias_good - mean_of_bias_nogood
                if np.max(delta) > 10 or np.min(delta) < -10:
                    print('''Obvious deviation between inside and outside of the hot era!!!
                    The result could be Unreliable!!!''')
                theta = 1.*THETA*n_good/(THETA*n_good+(1 - THETA)*n_nogood)
                mean_of_bias = theta * mean_of_bias_good + (1 - theta) * mean_of_bias_nogood

        if drawbias:
            chosen_n = random.randint(0, src_pts.shape[0] - 1)
            chosen_point = src_pts[chosen_n:chosen_n + 3, :]
            # chosen_point = src_pts
            self._drawbias(mat_file, img_arr, chosen_point, mean_of_bias, rec_boxes2, line_x2, delta_line_x)

        # print(chosen_point)
        # print(mean_of_bias)
        # print(chosen_point + mean_of_bias)
        # print(bias)
        return mean_of_bias.tolist(), delta_line_x


if __name__ == '__main__':
    IMG_DIC = './0408/86/20.0'
    m = Match()
    # img = cv2.imread('./0408/local_test/image.jpg')
    # m.save(img, './0408/local_test/img_base', line=False)
    # pt, des, kp = m.load('img')
    # m._drawkeypoints(img, kp)
    t0 = time.time()
    for imgpath in [i for i in walk(IMG_DIC) if i.endswith('.jpg')]:
    # for imgpath in ['./0408/86/0.0/image20160212011822.jpg']:
        print(imgpath)
        img = cv2.imread(imgpath)
        bias, delta_line_x = m.calc_bias('0408/local_test/img_base', img, drawbias=True, drawmatch=False,
                                         calc_line_bias=False)
        print('bias of pointmatch:', bias)
        print('bias of line match:', delta_line_x)
        print('time cost: {:.2f}ms'.format(1000 * (time.time() - t0)))
