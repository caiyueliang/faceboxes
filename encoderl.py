# encoding:utf-8

import torch
import math
import itertools
import cv2
import numpy as np


class DataEncoder:
    def __init__(self):
        '''
        compute default boxes
        '''
        scale = 1024.
        steps = [s / scale for s in (32, 64, 128)]      # [0.03125, 0.0625, 0.125]
        sizes = [s / scale for s in (32, 256, 512)]     # [0.03125, 0.25, 0.5]     当32改为64时，achor与label匹配的正样本数目更多
        aspect_ratios = ((1, 2, 4), (1,), (1,))
        feature_map_sizes = (32, 16, 8)

        density = [[-3, -1, 1, 3], [-1, 1], [0]]        # density for output layer1
        # density = [[0],[0],[0]] # density for output layer1

        num_layers = len(feature_map_sizes)             # 3
        boxes = []
        for i in range(num_layers):                     # 遍历3层中的每一层
            fmsize = feature_map_sizes[i]               # 分别为32, 16, 8
            # print(len(boxes))
            # 生成32×32个，16×16个, 8×8个二元组，如：(0,0), (0,1), (0,2), ... (1,0), (1,1), ..., (32,32)
            for h, w in itertools.product(range(fmsize), repeat=2):
                # print(h, w)
                cx = (w + 0.5)*steps[i]                     # 中心点坐标x
                cy = (h + 0.5)*steps[i]                     # 中心点坐标y

                s = sizes[i]
                for j, ar in enumerate(aspect_ratios[i]):
                    if i == 0:                          # 第1层加入检测框稠密策略
                        # j = (1, 2, 4)
                        # 16:(-3,-3),(-3,-1),(-3,1),(-3,3),(-1,-3),(-1,-1),(-1,1),(-1,3), ...
                        #  4:(-1,-1),(-1,1),(1,-1),(1,1), ...
                        #  1:(0,0)
                        for dx, dy in itertools.product(density[j], repeat=2):
                            boxes.append((cx+dx/8.*s*ar, cy+dy/8.*s*ar, s*ar, s*ar))
                    else:
                        # j = (1,), (1,)
                        boxes.append((cx, cy, s*ar, s*ar))

        self.default_boxes = torch.Tensor(boxes)
        print('default_boxes', self.default_boxes.size(), self.default_boxes * 1024)

    def test_iou(self):
        box1 = torch.Tensor([0, 0, 10, 10])
        box1 = box1[None, :]
        box2 = torch.Tensor([[5, 0, 15, 10], [5, 0, 15, 10]])
        print('iou', self.iou(box1, box2))

    def iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].

        Return:
          (tensor) iou, sized [N,M].
        '''

        N = box1.size(0)
        M = box2.size(0)
        print(box1.size(), box1)
        print(box2.size(), box2)

        lt = torch.max( # left top
            box1[:, :2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min( # right bottom
            box1[:, 2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def test_encode(self, boxes, img, label):
        # box = torch.Tensor([ 0.4003,0.0000,0.8409,0.4295])
        # box = box[None,:]
        # label = torch.LongTensor([1])
        # label = label[None,:]

        # 测试 encode
        loc, conf = self.encode(boxes, label, threshold=0.35)
        print('conf', type(conf), conf.size(), conf.long().sum())
        print('loc', type(loc), loc.size())
        # img = cv2.imread('test1.jpg')
        w, h, _ = img.shape

        # 画输入的boxes
        for box in boxes:
            cv2.rectangle(img, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0, 255, 0))

        # print(type(conf))
        for i in range(len(self.default_boxes)):
            if conf[i] != 0:
                print(i, conf[i])

        im = img.copy()
        # for i in range(42):
        #     print(self.default_boxes[i]*w)

        for i in range(32*32*21):
            box_item = self.default_boxes[i]*w
            # print(box_item)
            centerx, centery = int(box_item[0]), int(box_item[1])
            if conf[i] == 1:
                cv2.circle(im, (centerx, centery), 4, (0, 255, 0))      # 小绿圆，置信度不为0
            elif conf[i] == 2:
                cv2.circle(im, (centerx, centery), 4, (255, 0, 0))      # 小蓝圆，置信度不为0
            else:
                cv2.circle(im, (centerx, centery), 1, (0, 0, 255))      # 小红点
        box = self.default_boxes[0]
        cv2.rectangle(im, (0, 0), (int(box[2]*w), int(box[3]*h)), (0, 255, 0))
        box = self.default_boxes[16]
        cv2.rectangle(im, (0, 0), (int(box[2]*w), int(box[3]*h)), (0, 255, 0))
        box = self.default_boxes[20]
        cv2.rectangle(im, (0, 0), (int(box[2]*w), int(box[3]*h)), (0, 255, 0))
        cv2.imwrite('test_encoder_0.jpg', im)

        im = img.copy()
        for i in range(32*32*21, 32*32*21+16*16):
            box_item = self.default_boxes[i]*w
            centerx, centery = int(box_item[0]), int(box_item[1])
            if conf[i] == 1:
                cv2.circle(im, (centerx, centery), 4, (0, 255, 0))      # 小绿圆，置信度不为0
            elif conf[i] == 2:
                cv2.circle(im, (centerx, centery), 4, (255, 0, 0))      # 小蓝圆，置信度不为0
            else:
                cv2.circle(im, (centerx, centery), 2, (0, 0, 255))      # 小红点
        box = self.default_boxes[32*32*21]
        cv2.rectangle(im, (0, 0), (int(box[2]*w), int(box[3]*w)), (0, 255, 0))
        cv2.imwrite('test_encoder_1.jpg', im)

        im = img.copy()
        for i in range(32*32*21+16*16, len(self.default_boxes)):
            box_item = self.default_boxes[i]*w
            centerx, centery = int(box_item[0]), int(box_item[1])
            if conf[i] == 1:
                cv2.circle(im, (centerx, centery), 4, (0, 255, 0))      # 小绿圆，置信度不为0
            elif conf[i] == 2:
                cv2.circle(im, (centerx, centery), 4, (255, 0, 0))      # 小蓝圆，置信度不为0
            else:
                cv2.circle(im, (centerx, centery), 2, (0, 0, 255))      # 小红点
        box = self.default_boxes[32*32*21+16*16]
        cv2.rectangle(im, (0, 0), (int(box[2]*w), int(box[3]*w)), (0, 255, 0))
        cv2.imwrite('test_encoder_2.jpg', im)

        # for i in range(conf.size(0)):
            # if conf[i].numpy != 0:
                # print()

        # 测试 decode
        conf_change = []
        for c in conf:
            if c == 0:
                conf_change.append([1, 0, 0])
            elif c == 1:
                conf_change.append([0, 1, 0])
            else:
                conf_change.append([0, 0, 1])
        boxes, labels, max_conf = self.decode(loc, torch.Tensor(conf_change), False)
        print(boxes, labels, max_conf)

        im = img.copy()
        for box in boxes:
            cv2.rectangle(im, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (255, 255, 0))
        cv2.imwrite('test_encoder_3.jpg', im)

    def encode(self, boxes, classes, threshold=0.35):
        '''
        boxes:          [num_obj, 4]                    # 二维 [[x1, y1, x2, y2], [x3, y3, x4, y4]]
            default_box (x1,y1,x2,y2)
        classes:class   label [obj,]                    # [1, 1]
        return:boxes:   (tensor) [num_obj,21824,4]
        '''
        boxes_org = boxes
        print('boxes', boxes.size(), boxes)
        print('classes', classes.size(), classes)

        default_boxes = self.default_boxes              # [21824, 4]
        num_default_boxes = default_boxes.size(0)       # 21824
        num_obj = boxes.size(0)                         # 人脸个数
        print('[encode] num_obj', num_obj)

        # 计算iou,交并比
        iou = self.iou(
            boxes,
            torch.cat([default_boxes[:, :2]-default_boxes[:, 2:]/2, default_boxes[:, :2]+default_boxes[:, 2:]/2], 1))
        # print('iou ', iou.size(), iou)
        # for i, iou_line in enumerate(iou):
        #     for j, iou_item in enumerate(iou_line):
        #         if iou_item > threshold:
        #             print(i, j, iou_item)

        max_iou, max_iou_index = iou.max(1)         # 为boxes中的每一个bounding box（边界框），不管IOU大小，都设置一个与之IOU最大的default_box
        print('max_iou', max_iou)                   # max_iou, default_box中，IOU最大的值是多少;有几个边界框，就有几个最大值
        print('max_iou_index', max_iou_index)       # max_iou_index, default_box中，IOU最大的值对应的index;有几个边界框，就有几个索引值

        iou, max_index = iou.max(0)                 # 每一个default_boxes对应到与之IOU最大的boxes上的bounding box（边界框）
        print('iou', iou.size(), iou)
        print('max_index', max_index.size(), max_index)

        # print(max(iou))
        max_index.squeeze_(0)                           # torch.LongTensor 21824
        iou.squeeze_(0)
        print('max_index', max_index.size(), max_index)
        print('iou', iou.size(), iou)

        print('max_index[max_iou_index]', max_index[max_iou_index])
        print("torch.LongTensor(range(num_obj))", torch.LongTensor(range(num_obj)))
        max_index[max_iou_index] = torch.LongTensor(range(num_obj))
        print('max_index[max_iou_index]', max_index[max_iou_index])
        print('max_index', max_index.size(), max_index)

        boxes = boxes[max_index]                        # [21824,4]是图像label
        print('boxes', boxes.size(), boxes)

        variances = [0.1, 0.2]
        cxcy = (boxes[:, :2] + boxes[:, 2:])/2 - default_boxes[:, :2]       # [21824,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]           # [21824,2]  为什么会出现0宽度？？
        wh = torch.log(wh) / variances[1]                                   # Variable. log求自然对数

        inf_flag = wh.abs() > 10000

        # print('inf_flag.long().sum()', inf_flag.long().sum().item())
        # if inf_flag.long().sum() is not 0:
        if inf_flag.long().sum().item() is not 0:
            print('inf_flag has true', wh, boxes)
            print('org_boxes', boxes_org)
            print('max_iou', max_iou, 'max_iou_index', max_iou_index)
            raise BaseException('[my exception] inf error')

        loc = torch.cat([cxcy, wh], 1)      # [21824,4]
        print('loc', loc.size(), loc)

        conf = classes[max_index]           # 其实都是1 [21824,]
        print('conf', conf.size(), conf)
        conf[iou < threshold] = 0           # iou小的设为背景， 0为背景
        print('conf', conf.size(), conf)
        # conf[max_iou_index] = 1             # 这么设置有问题，loc loss 会导致有inf loss，从而干扰训练，
        # print('conf', conf.size(), conf)
                                            # 去掉后，损失降的更稳定些，是因为widerFace数据集里有的label
                                            # 做的宽度为0，但是没有被滤掉，是因为max(1)必须为每一个object选择一个
                                            # 与之对应的default_box，需要修改数据集里的label。

        # ('targets', Variable containing:
        # 318.7500   -1.2500      -inf      -inf
        # org_boxes 0.1338  0.3801  0.1338  0.3801

        return loc, conf

    def nms(self, bboxes, scores, threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1) * (y2-y1)

        _,order = scores.sort(0,descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i].item())
            yy1 = y1[order[1:]].clamp(min=y1[i].item())
            xx2 = x2[order[1:]].clamp(max=x2[i].item())
            yy2 = y2[order[1:]].clamp(max=y2[i].item())

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)

    def decode(self, loc, conf, use_gpu, nms_threshold=0.5):
        '''
        將预测出的 loc/conf转换成真实的人脸框
        loc [21842, 4]
        conf [21824, 2]
        '''
        print('loc', loc.size(), loc)
        print('conf', conf.size(), conf)
        variances = [0.1, 0.2]

        # variances = [0.1, 0.2]
        # cxcy = (boxes[:, :2] + boxes[:, 2:])/2 - default_boxes[:, :2]       # [21824,2]
        # cxcy /= variances[0] * default_boxes[:, 2:]
        # wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]           # [21824,2]  为什么会出现0宽度？？
        # wh = torch.log(wh) / variances[1]                                   # Variable. log求自然对数
        if use_gpu:
            cxcy = loc[:, :2].cuda() * variances[0] * self.default_boxes[:, 2:].cuda() + self.default_boxes[:, :2].cuda()
            wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:].cuda()
            boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)                        # [21824,4]
        else:
            cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
            wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]   # 返回一个新张量，包含输入input张量每个元素的指数
            boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)                        # [21824,4]

        print('cxcy', cxcy.size(), cxcy)
        print('wh', wh.size(), wh)
        print('boxes', boxes.size(), boxes)

        conf[:, 0] = 0.4    # 置信度第0列（背景）设为0.4，下面再取最大值，目的是为了过滤置信度小于0.4的标签
        max_conf, labels = conf.max(1)                          # [21842,1]

        # print(max_conf)
        # print('labels', labels.long().sum().item())
        if labels.long().sum().item() is 0:                     # 标签和为0？表示图片没有标签？
            sconf, slabel = conf.max(0)
            max_conf[slabel[0:5]] = sconf[0:5]
            labels[slabel[0:5]] = 1

        # print('labels', labels)
        ids = labels.nonzero().squeeze(1)
        print('ids', ids)
        # print('boxes', boxes.size(), boxes[ids])

        keep = self.nms(boxes[ids], max_conf[ids], nms_threshold)   # .squeeze(1))

        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]


if __name__ == '__main__':
    dataencoder = DataEncoder()
    # dataencoder.test_iou()

    # img = cv2.imread("Data/9488513_鄂A578U2_3.jpg")
    # h, w, _ = img.shape
    # img = cv2.resize(img, (1024, 1024))
    # dataencoder.test_encode(torch.Tensor([[32./w, 266./h, 262./w, 351./h], [455./w, 138./h, 572./w, 179./h]]), img, torch.LongTensor([1, 1]))

    img = cv2.imread("Data/557831_蓝_粤YXU501.jpg")
    h, w, _ = img.shape
    img = cv2.resize(img, (1024, 1024))
    # 12.0 15.0 231.0 110.0 1
    # 1072.0 591.0 239.0 112.0 1
    # 1461.0 275.0 94.0 36.0 1
    # 414.0 5.0 1092.0 818.0 2
    # 1185.0 16.0 435.0 352.0 2
    boxes = torch.Tensor([[12. / w, 15. / h, 243. / w, 125. / h],
                          [1072. / w, 591. / h, 1311. / w, 703. / h],
                          [1461. / w, 275. / h, 1565. / w, 311. / h],
                          [414. / w, 5. / h, 1506. / w, 823. / h],
                          [1185. / w, 16. / h, 1620. / w, 368. / h]])
    dataencoder.test_encode(boxes, img, torch.LongTensor([1, 1, 1, 2, 2]))

