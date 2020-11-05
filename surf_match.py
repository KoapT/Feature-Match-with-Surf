#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Editor      : PyCharm
#   File name   : surf_match.py
#   Author      : Koap
#   Created date: 2020/3/4 上午11:00
#   Description : opencv-python=3.4.3.18 opencv-contrib-python=3.4.1.15
#
# ================================================================

import cv2
import numpy as np
import time
import os
import random
import rospy

### configs:
PROPORTION = [.05, .5]  # 检测矩形框区域相对整个图案的最小、最大占比
ASPECT_RATIO = [.8, 2.5]  # 检测矩形框区域的长宽比范围
KERNEL_morphologyEx = [8, 8]  # 形态学变换的核
TEMPLATE_THERSH = .6  # 模板匹配阈值，越大匹配精度要求越高
MIN_KP_SURF = 50  # 小于该值使用单通道做SURF


###

def walk(dir_path):
    file_list = []
    for (root, dirs, files) in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def templatematch(template, img_arr,channel=3 ,threshold=TEMPLATE_THERSH):
    '''
    模板匹配。对每个模板划分左中右三等份之后分别匹配。
    :param template:模板，np.ndarray
    :param img_arr: 待匹配的图片
    :param threshold: 匹配的阈值，阈值越高，要求越高，匹配越精确。
    :return:(是否匹配成功， 匹配到的四边形的坐标)
    '''
    if channel==3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    elif channel in [0,1,2]:
        template = template[:,:,channel]
        img_gray = img_arr[:,:,channel]
    else:
        return False, []
    H, W = template.shape
    coords = []
    try:
        templates = [template, template[:, :int(W / 2 * .7)], template[:, int(W / 2 * 1.2):]]
        for temp in templates:
            h, w = temp.shape[:2]
            res = cv2.matchTemplate(img_gray, temp, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            if (res >= threshold).any():
                coord = np.array(list(zip(*loc[::-1])))
                mean_coord = outlier_filtering(coord)
                mean_coord = np.concatenate([mean_coord, mean_coord + np.array([w, h])], axis=0)
                coords.append(mean_coord)
    except:
        return False, coords
    if coords:
        return True, coords
    return False, coords


def outlier_filtering(bias, axis=0):
    '''
    基于均值和标准差的离群值过滤算法。
    :param bias: 待过滤的值
    :param axis: np.array的轴
    :return:过滤离群值后的均值
    '''
    mean_of_bias = np.mean(bias, axis=axis)
    std_of_bias = np.std(bias, axis=axis)
    mask_max = bias <= (mean_of_bias + std_of_bias)
    mask_min = bias >= (mean_of_bias - std_of_bias)
    mask_ = np.logical_and(
        np.logical_and(mask_max[:, 0], mask_max[:, 1]),
        np.logical_and(mask_min[:, 0], mask_min[:, 1])
    )
    bias = bias[mask_, :]
    mean_of_bias = np.mean(bias, axis=0)
    return mean_of_bias


class Match(object):
    def __init__(self, debug=False):
        super(Match, self).__init__()
        self.debug_mode = debug
        self.rec_boxes1 = np.array([[0, 0, 0, 0]], dtype=np.int32)
        self.rec_boxes2 = np.array([[0, 0, 0, 0]], dtype=np.int32)
        self.now_time = ''

    def _surf(self, img_arr):
        surf = cv2.xfeatures2d_SURF.create()  #can not be used with opencv3.4.3 or later 
        keyPoint, descriptor = surf.detectAndCompute(img_arr, mask=None)  # 特征提取得到关键点以及对应的特征向量
        return keyPoint, descriptor

    def findRectangle(self, img_arr):
        '''
        :return: a list
        '''
        rec_boxes = []
        H, W, _ = img_arr.shape
        img_area = H * W
        imgray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        thresh, img_thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones(KERNEL_morphologyEx, np.uint8)
        img_opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        img_edge = cv2.Canny(img_opening, 40, 20)
        image, contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = box = cv2.boundingRect(contour)
            if img_area * PROPORTION[0] < w * h < img_area * PROPORTION[1] and \
                    ASPECT_RATIO[0] < 1.0 * w / h < ASPECT_RATIO[1]:
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
            x1, y1, x2, y2 = rec
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def save(self, img_arr, save_name):
        '''
        保存预置位信息，保存预置位的图片.jpg 和 特征信息.npz
        :param img_arr: 读取的图片
        :param save_name:  保存的路径（含文件名，不含后缀名） type->src
        :return  保存成功时返回True，
                 保存失败时返回False
        '''
        if img_arr.ndim != 3:
            return False

        kp, des = self._surf(img_arr)
        num_kp = len(kp)
        channel = 3
        if num_kp < MIN_KP_SURF:
            for i in range(3):
                kpi, desi = self._surf(img_arr[:, :, i])
                if len(kpi) > num_kp:
                    num_kp = len(kpi)
                    channel = i
                    kp = kpi
                    des = desi
        pt = np.float32([m.pt for m in kp]).reshape(-1, 1, 2)

        try:
            cv2.imwrite(save_name + '.jpg', img_arr)
            np.savez(save_name, pt=pt, des=des, chn=channel)
            return True
        except IOError:
            if self.debug_mode:
                print('Error: 保存路径错误！')
            else:
                rospy.loginfo('Error: 保存路径错误！')
            return False
        except:
            return False

    def load(self, mat_file):
        '''
        读取预置位信息
        :param mat_file: 读取的路径（含文件名，不含后缀名）
        :return  读取成功时返回tuple(pt,des,kps,rec_boxes)，
                 读取失败时返回None
        '''

        try:
            arr = np.load(mat_file + '.npz', allow_pickle=True)
        except:
            if self.debug_mode:
                print('Error: 读取路径错误！没有找到文件%s.npz' % mat_file)
            else:
                rospy.loginfo('Error: 读取路径错误！没有找到文件%s.npz' % mat_file)
            return None

        try:
            with open(mat_file + '.mark', 'r') as f:
                temp = f.readline().strip()
                temp_coord = list(map(lambda x: int(x), temp.split(',')))
        except:
            if self.debug_mode:
                print('Error: 读取路径错误！没有找到文件%s.mark' % mat_file)
            else:
                rospy.loginfo('Error: 读取路径错误！没有找到文件%s.mark' % mat_file)
            return None

        pt, des, chn = arr['pt'], arr['des'], arr['chn']
        kps = []
        for i in pt:
            kp = cv2.KeyPoint()
            kp.pt = tuple(i[0])
            kps.append(kp)
        return chn, pt, des, kps, temp_coord

    def _drawkeypoints(self, img_arr, kp):
        img = cv2.drawKeypoints(img_arr, kp, None)
        if self.debug_mode:
            cv2.imshow('keypoints', img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def _drawmatch(self, name, mat_file, kp1, img, kp2, matches, matchesMask):
        img1 = cv2.imread(mat_file + '.jpg')
        img2 = img.copy()
        self._drawRectangle(img1, self.rec_boxes1)
        if len(self.rec_boxes2) > 0:
            self._drawRectangle(img2, self.rec_boxes2)
        draw_params = dict(
            matchesMask=matchesMask,
            flags=2)  # draw only inliers
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, **draw_params)
        if self.debug_mode:
            cv2.imshow(name, img)
        else:
            cv2.imwrite(mat_file + '_match_%s.jpg' % self.now_time, img)

    def _drawbias(self, mat_file, img_arr, chosen_point, mean_of_bias):
        img1 = cv2.imread(mat_file + '.jpg')
        self._drawRectangle(img1, self.rec_boxes1)
        img2 = img_arr.copy()
        if len(self.rec_boxes2) > 0:
            self._drawRectangle(img2, self.rec_boxes2)

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
        cv2.putText(img, text1, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        if self.debug_mode:
            cv2.imshow("pointmatch", img)
        else:
            cv2.imwrite(mat_file + '_bias_%s.jpg' % self.now_time, img)

    def _ransac(self, src_pts_all, dst_pts_all, threshold, method=0):
        if src_pts_all.ndim == 2:
            src_pts_all = src_pts_all[:, None, :]
        if dst_pts_all.ndim == 2:
            dst_pts_all = dst_pts_all[:, None, :]
        if method == 1:
            M, mask = cv2.findHomography(src_pts_all, dst_pts_all, cv2.RANSAC, threshold)
        else:
            M, mask = cv2.findFundamentalMat(src_pts_all, dst_pts_all, cv2.RANSAC, threshold)
        return mask

    def calc_bias(self, mat_file, img_arr, drawbias=False, drawmatch=False, use_final_track=True):
        '''
        计算预置位和当前图像的位移
        计算流程说明：
        1）模板匹配
        2）SURF+BFMATCH特征点匹配
        3）全局RANSAC1筛选匹配对
        4）根据矩形范围reclimit1筛选匹配对
        5）根据矩形范围reclimit2筛选匹配对
        6）局部RANSAC2筛选匹配对
        7）根据最终筛选出的匹配对计算位移偏差的像素值
        以上每一步不满足条件都会导致匹配失败。
        :param mat_file: 预置位文件的路径（含文件名，不含后缀名）
        :param img_arr： 当前位置的图像
        :param drawbias： 根据计算得到的位移像素值，画出当前图像和预置位图像的对比图，供验证查看。
        :param drawmatch： 画出当前图像和预置位图像所有匹配的特征点和对应信息。
        :return 2部分分别是：
                1.result: 当前图像相对于预置位图像水平位移的像素值，正数表示当前图像相对预置位图像右移。 若没有匹配点，则返回0
                2.ret: 1 or 0, 1 时可信， 0不可信

        '''
        if img_arr.ndim != 3:
            if self.debug_mode:
                print('当前图片读取失败')
            else:
                rospy.loginfo('当前图片读取失败')
            return 0, 0
        self.now_time = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime(time.time()))

        if drawbias and (not self.debug_mode):
            try:
                cv2.imwrite(mat_file + '_' + self.now_time + '.jpg', img_arr)
            except:
                rospy.loginfo('当前图片保存失败')

        loadresult = self.load(mat_file)
        if loadresult == None:
            if self.debug_mode:
                print('预置位读取失败')
            else:
                rospy.loginfo('预置位读取失败')
            return 0, 0
        chn, pt1, des1, kp1, tempcoord = loadresult

        c = tempcoord
        template = cv2.imread(mat_file + '.jpg')[c[1]:c[3], c[0]:c[2],:]

        ret, coords = templatematch(template, img_arr, channel=chn)

        if not ret:
            if self.debug_mode:
                print('没有匹配到模板！')
            else:
                rospy.loginfo('没有匹配到模板！')
            return 0, 0
        self.rec_boxes1 = np.array([tempcoord])
        self.rec_boxes2 = np.array(coords, dtype=np.int32)

        if chn == 3:
            kp2, des2 = self._surf(img_arr)
        elif chn in [0,1,2]:
            kp2, des2 = self._surf(img_arr[:, :, chn])
        else:
            if self.debug_mode:
                print('错误的channel值！')
            else:
                rospy.loginfo('错误的channel值！')
            return 0, 0
        pt2 = np.float32([m.pt for m in kp2]).reshape(-1, 1, 2)

        # 显示surf匹配到的点
        # self._drawkeypoints(img_arr,kp2)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)

        src_pts = pt1[[m.queryIdx for m in matches], :, :]
        dst_pts = pt2[[m.trainIdx for m in matches], :, :]
        bias = dst_pts - src_pts

        def merge_mask(mask, mask1, name):
            mask *= mask1
            l_mask = mask.ravel().tolist()
            if drawmatch:
                self._drawmatch(name, mat_file, kp1, img_arr, kp2, matches, l_mask)
            return mask, l_mask

        # RANSAC
        mask_ransac1 = self._ransac(src_pts, dst_pts, threshold=30, method=0)  # 此处threshold可以设得大一点。
        if mask_ransac1 is None:
            if self.debug_mode:
                print('第1次ransac，没找到匹配点')
            else:
                rospy.loginfo('第1次ransac，没找到匹配点')
            return 0, 0
        mask = np.ones_like(mask_ransac1)
        mask, l_mask = merge_mask(mask, mask_ransac1, 'ransac1')
        num_rested_points = mask_ransac1[mask_ransac1 == 1].shape[0]
        if num_rested_points < 12:
            if self.debug_mode:
                print('第1次ransac，没找到足够多的匹配点([%d]<12)' % num_rested_points)
            else:
                rospy.loginfo('第1次ransac，没找到足够多的匹配点([%d]<12)' % num_rested_points)
            return 0, 0

        # TEMPLATE REC LIMIT
        x1, y1, x1_, y1_ = self.rec_boxes1[0]
        mask_rec_x1 = np.logical_and(x1 <= src_pts[..., 0], src_pts[..., 0] <= x1_)
        mask_rec_y1 = np.logical_and(y1 <= src_pts[..., 1], src_pts[..., 1] <= y1_)
        mask_rec_1 = np.logical_and(mask_rec_x1, mask_rec_y1)
        mask_rec_1 = mask_rec_1.astype(np.uint8)
        mask, l_mask = merge_mask(mask, mask_rec_1, 'recLimit1')
        if not any(l_mask):
            if self.debug_mode:
                print('预置位选取区域中，没找到匹配点')
            else:
                rospy.loginfo('预置位选取区域中，没找到匹配点')
            return 0, 0

        # TEMPLATE REC2 LIMIT
        mask_recs_2 = []
        for rec2 in self.rec_boxes2:
            x2, y2, x2_, y2_ = rec2
            mask_rec_x2 = np.logical_and(x2 <= dst_pts[..., 0], dst_pts[..., 0] <= x2_)
            mask_rec_y2 = np.logical_and(y2 <= dst_pts[..., 1], dst_pts[..., 1] <= y2_)
            mask_rec_2 = np.logical_and(mask_rec_x2, mask_rec_y2)
            mask_recs_2.append(mask_rec_2)
        mask_rec_2 = np.max(np.array(mask_recs_2), axis=0)
        mask, l_mask = merge_mask(mask, mask_rec_2, 'recLimit2')
        if not any(l_mask):
            if self.debug_mode:
                print('模板匹配选取的区域中，没找到匹配的点')
            else:
                rospy.loginfo('模板匹配选取的区域中，没找到匹配的点')
            return 0, 0

        # RANSAC2
        mask_bool = mask.astype(np.bool)

        matches = np.array(matches)
        matches = matches[mask_bool.ravel()]
        matches = matches.tolist()

        src_pts = src_pts[mask_bool]
        dst_pts = dst_pts[mask_bool]
        bias = bias[mask_bool]

        if src_pts.shape[0] < 12:
            if self.debug_mode:
                print('模板区域内的匹配点过少,仅%d个！' % src_pts.shape[0])
            else:
                rospy.loginfo('模板区域内匹配点过少,仅%d个！' % src_pts.shape[0])
            return 0, 0

        mask_ransac2 = self._ransac(src_pts, dst_pts, 2, method=1)
        if mask_ransac2 is None:
            if self.debug_mode:
                print('第2次ransac，没找到匹配点')
            else:
                rospy.loginfo('第2次ransac，没找到匹配点')
            return 0, 0

        mask = mask_ransac2
        mask_bool = mask.astype(np.bool)[:, 0]
        l_mask = mask.ravel().tolist()
        if drawmatch:
            self._drawmatch('ransac2', mat_file, kp1, img_arr, kp2, matches, l_mask)
        num_rested_points = mask_ransac2[mask_ransac2 == 1].shape[0]
        if num_rested_points < 12:
            if self.debug_mode:
                print('第2次ransac，没找到足够多的匹配点([%d]<12)' % num_rested_points)
            else:
                rospy.loginfo('第2次ransac，没找到足够多的匹配点([%d]<12)' % num_rested_points)
                
            #return 0, 0   
            #yhy
            if use_final_track:
                try:
                    print('use final track match')
                    print('cv2 version: {}'.format(cv2.__version__))
                    if self.debug_mode:
                        print('use final track match')
                        print('cv2 version: {}'.format(cv2.__version__))
                    else:
                        rospy.loginfo('use final track match')
                        rospy.loginfo('cv2 version: {}'.format(cv2.__version__))
                    img_template = cv2.imread(mat_file + '.jpg')
                    h = img_template.shape[0]
                    w = img_template.shape[1]
                    margin = 0
                    left = tempcoord[0] - margin
                    top = tempcoord[1] - margin
                    right = tempcoord[2] + margin
                    bottom = tempcoord[3] + margin
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    if right >= w:
                        right = w - 1
                    if bottom >= h:
                        bottom = h - 1
                    if left >= right or top >= bottom:
                        if self.debug_mode:
                            print('track box erro')
                        else:
                            rospy.loginfo('track box erro')
                        return 0, 0
                    bbox_template = (left, top, right-left, bottom-top)
                    tracker = cv2.TrackerCSRT_create()#TrackerKCF_create()#  #TrackerCSRT_create in opencv-contrib-python 3.4.1.15 or later
                    tracker.init(img_template, bbox_template)
                    success, bbox = tracker.update(img_arr)
                    wcenter_template = bbox_template[0] + bbox_template[2] / 2.0
                    hcenter_template = bbox_template[1] + bbox_template[3] / 2.0
                    wcenter = bbox[0] + bbox[2] / 2.0
                    hcenter = bbox[1] + bbox[3] / 2.0
                    cv2.rectangle(
                            img_template, 
                            (int(bbox_template[0]), int(bbox_template[1])), 
                            (int(bbox_template[0]+bbox_template[2]), int(bbox_template[1]+bbox_template[3])), 
                            (0, 255, 0), 2)
                    cv2.circle(
                            img_template, 
                            (int(wcenter_template), int(hcenter_template)), 3, (0, 255, 0), 2)
                    if self.debug_mode:
                        print('track success:{}'.format(success))
                    else:
                        rospy.loginfo('track success:{}'.format(success))
                    if success:
                        bias = wcenter - wcenter_template
                        cv2.putText(
                                img_template, 'bias of trackmatch:{}'.format(bias), 
                                (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
                        cv2.rectangle(
                            img_arr, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), 
                            (255, 0, 0), 2)
                        cv2.circle(
                                img_arr, 
                                (int(wcenter), int(hcenter)), 3, (255, 0, 0), 2)
                        img_saved = cv2.hconcat([img_template, img_arr])
                        if self.debug_mode:
                            cv2.imshow("trackmatch", img_saved)
                        else:
                            cv2.imwrite(mat_file + '_trackbias_%s.jpg' % self.now_time, img_saved)
                        print('track match success')
                        return bias, 1
                    else:
                        return 0, 0
                except Exception as e:
                    rospy.loginfo('track match erro: {}'.format(e))
                    print('track match erro: {}'.format(e))
                    return 0, 0
            else:
                return 0, 0

        # 计算位移
        bias = bias[mask_bool]
        mean_of_bias = np.mean(bias, axis=0)

        chosen_n = random.randint(0, src_pts.shape[0] - 1)
        chosen_point = src_pts[chosen_n:chosen_n + 9, :]
        if drawbias:
            self._drawbias(mat_file, img_arr, chosen_point, mean_of_bias)

        if self.debug_mode:
            print('成功！')
        else:
            rospy.loginfo('成功！')

        return mean_of_bias[0], 1


# local test
if __name__ == '__main__':
    print('start test')
    IMG_DIC = os.getcwd()
    PRESET_PATH = os.path.join(IMG_DIC, 'image')

    m = Match(debug=False)
    # img = cv2.imread('/home/tk/projects/Feature-Match-with-Surf/0408/local_test/image.jpg')
    # print(m.save(img, '/home/tk/projects/Feature-Match-with-Surf/0408/local_test/image'))
    # pt, des, kp, rec, tempcoord = m.load('./0408/local_test/img_base')
    # print(rec, '\n', tempcoord)
    # m._drawkeypoints(img, kp)
#    img_template = cv2.imread(PRESET_PATH + '.jpg')
#    m.save(img_template, 'image')
    for imgpath in [i for i in walk(IMG_DIC) if i.endswith('.jpg')]:
        # for imgpath in ['/media/tk/DATA1/轨道机器人/0703/data0730/149/218.7/image_20200730_02h47m57s.jpg']:
        t0 = time.time()
        print('test: {}'.format(imgpath))
        img = cv2.imread(imgpath)
        #cv2.imshow('present picture', img)
        bias, ret = m.calc_bias(PRESET_PATH, img, drawbias=True, drawmatch=True)
        print('test done, bias={}, ret={}'.format(bias, ret))
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
