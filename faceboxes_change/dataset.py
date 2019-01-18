# encoding:utf-8
'''
txt描述文件 image_name.jpg num x y w h 1 x y w h 1 这样就是说一张图片中有两个人脸
'''
import os
import sys
import os.path

import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import torch.utils.data as data

import cv2

from encoderl import DataEncoder


class ListDataset(data.Dataset):
    image_size = 1024

    def __init__(self, root, list_file, train, transform):
        print('data init')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []                                # 文件名列表
        self.boxes = []                                 # 位置标签列表，格式 [x1, y1, x2, y2]
        self.labels = []                                # 类别标签列表?
        self.small_threshold = 10. / self.image_size    # face that small than threshold will be ignored
        self.data_encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            # print('splited', splited)
            self.fnames.append(splited[0])
            num_faces = int(splited[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(splited[2+5*i])
                y = float(splited[3+5*i])
                w = float(splited[4+5*i])
                h = float(splited[5+5*i])
                c = int(splited[6+5*i])
                box.append([x, y, x+w, y+h])            # [[x1, y1, x2, y2], [x3, y3, x4, y4]]
                label.append(c)                         # [1, 1]
                # print('box', box, 'label', label)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        # print(os.path.join(self.root + fname))
        assert img is not None

        boxes = self.boxes[idx].clone()         # 获取某一张图片对应的位置信息（可能多个）
        labels = self.labels[idx].clone()       # 获取某一张图片对应的类别信息（可能多个）

        # 图片增广
        if self.train:
            img, boxes, labels = self.random_crop(img, boxes, labels)       # 随机裁剪
            img = self.random_bright(img)                                   # 随机调亮
            # img, boxes = self.random_flip(img, boxes)                     # 随机翻转
            # boxwh = boxes[:, 2:] - boxes[:, :2]
            # print('boxwh', boxwh)

        h, w, _ = img.shape
        img = cv2.resize(img, (self.image_size, self.image_size))

        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)                # 位置信息除以宽或高（归一化）
        # for t in self.transform:
        #     img = t(img)
        img = self.transform(img)

        loc_target, conf_target = self.data_encoder.encode(boxes, labels)   # 对位置标签进行转换

        return img, loc_target, conf_target

    def __len__(self):
        return self.num_samples

    def random_getim(self):
        idx = random.randrange(0,self.num_samples)
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root+fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        return img, boxes, labels

    # 随机翻转
    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    # # 随机裁剪
    # def random_crop(self, img, boxes, labels, prob=0.8):
    #     # print('random_crop', boxes, labels)
    #     h, w, _ = img.shape
    #
    #     # show_img = img.copy()
    #     # for box in boxes:
    #     #     cv2.rectangle(show_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
    #     # cv2.imshow('old_image', show_img)
    #     # cv2.waitKey(0)
    #
    #     x1 = boxes[:, 0]
    #     y1 = boxes[:, 1]
    #     x2 = boxes[:, 2]
    #     y2 = boxes[:, 3]
    #     # print('x1, y1, x2, y2', x1, y1, x2, y2)
    #
    #     min_left = min(min(x1), min(x2))
    #     min_top = min(min(y1), min(y2))
    #     min_lt = min(min_left, min_top)
    #
    #     min_right = w - max(max(x1), max(x2))
    #     min_bottom = h - max(max(y1), max(y2))
    #     min_rb = min(min_right, min_bottom)
    #     # print('min_left, min_top, min_right, min_bottom', min_left, min_top, min_right, min_bottom)
    #
    #     crop_left = 0
    #     crop_top = 0
    #     crop_right = w
    #     crop_bottom = h
    #
    #     # random crop left and top
    #     if random.random() < prob:
    #         rate = random.random()
    #         crop = int(min_lt * rate)
    #
    #         x1 = x1 - crop
    #         x2 = x2 - crop
    #         crop_left = crop
    #         # print('crop_left', crop_left, rate, x1, x2)
    #
    #         y1 = y1 - crop
    #         y2 = y2 - crop
    #         crop_top = crop
    #         # print('crop_top', crop_top, rate, y1, y2)
    #
    #     # random crop right
    #     if random.random() < prob:
    #         rate = random.random()
    #         crop = int(min_rb * rate)
    #
    #         crop_right = crop_right - crop
    #         # print('crop_right', crop_right, rate)
    #
    #         crop_bottom = crop_bottom - crop
    #         # print('crop_bottom', crop_bottom, rate)
    #
    #     img = img[crop_top:crop_bottom, crop_left:crop_right]
    #
    #     boxes[:, 0] = x1    # 第1列表示：x轴中心点（比例） # 第0列表示：类别
    #     boxes[:, 1] = y1    # 第2列表示：y轴中心点（比例）
    #     boxes[:, 2] = x2    # 第3列表示：w（比例）
    #     boxes[:, 3] = y2    # 第4列表示：h（比例）
    #     # print('new labels', labels)
    #
    #     # show_img = img.copy()
    #     # for box in boxes:
    #     #     cv2.rectangle(show_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
    #     # cv2.imshow('new_image', show_img)
    #     # cv2.waitKey(0)
    #
    #     return img, boxes, labels

    # 随机裁剪
    def random_crop(self, im, boxes, labels):
        cv2.imshow('old_image', im)
        print('random_crop', boxes, labels)

        # show_img = im.copy()
        # for box in boxes:
        #     cv2.rectangle(show_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
        # cv2.imshow('old_image', show_img)

        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        # print(imh, imw, short_size)
        while True:
            mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
            print('mode', mode)
            if mode is None:
                boxes_uniform = boxes / torch.Tensor([imw, imh, imw, imh]).expand_as(boxes)
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)
                if not mask.any():
                    print('default image have none box bigger than small_threshold')
                    im, boxes, labels = self.random_getim()
                    imh, imw, _ = im.shape
                    short_size = min(imw, imh)
                    continue
                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))

                # show_img = im.copy()
                # for box in selected_boxes:
                #     cv2.rectangle(show_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
                # cv2.imshow('new_image', show_img)
                # cv2.waitKey(0)

                return im, selected_boxes, selected_labels

            for _ in range(10):
                # rate = random.choice([, 0.3, 0.5, 0.7, 0.9])
                w = random.randrange(int(mode*short_size), short_size)
                h = w

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                roi2 = roi.expand(len(center), 4)
                mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])
                mask = mask[:, 0] & mask[:, 1]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                img = im[y:y+h, x:x+w, :]
                selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)
                # print('croped')

                boxes_uniform = selected_boxes / torch.Tensor([w, h, w, h]).expand_as(selected_boxes)
                print('boxes_uniform', boxes_uniform)
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                print('boxwh', boxwh)
                mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)
                print('mask', mask)

                if not mask.any():
                    print('crop image have none box bigger than small_threshold')
                    im, boxes, labels = self.random_getim()
                    imh, imw, _ = im.shape
                    short_size = min(imw, imh)
                    continue
                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                print('selected_boxes_selected', selected_boxes_selected)
                print('selected_labels', selected_labels)

                # show_img = img.copy()
                # for box in selected_boxes_selected:
                #     cv2.rectangle(show_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
                # cv2.imshow('new_image', show_img)
                # cv2.waitKey(0)

                return img, selected_boxes_selected, selected_labels

    # 随机调亮
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im

    def testGet(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root,fname))
        cv2.imwrite('test_encoder_source.jpg', img)
        boxes = self.boxes[idx].clone()
        # print(boxes)
        labels = self.labels[idx].clone()

        for box in boxes:
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255))
        cv2.imwrite(fname, img)

        if self.train:
            img, boxes, labels = self.random_crop(img, boxes, labels)
            img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes)

        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        img = cv2.resize(img,(self.image_size,self.image_size))
        for t in self.transform:
            img = t(img)

        print(idx, fname, boxes)

        return img, boxes, labels


if __name__ == '__main__':
    transform_train = T.Compose([
        # T.Resize((self.img_size, self.img_size)),
        T.ToTensor(),
        # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

    file_root = '../../Data/yolo/yolo_data_new/car_detect_train/'
    train_label = os.path.join(file_root, 'faceboxes_label.txt')
    train_dataset = ListDataset(root=file_root, list_file=train_label, train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=1)

    for i, (images, loc_targets, conf_targets) in enumerate(train_loader):
        print(i)

    # train_dataset = ListDataset(root=file_root, list_file='image_path.txt', train=True, transform=[transforms.ToTensor()])
    # print('the dataset has %d image' % (len(train_dataset)))
    # for i in range(len(train_dataset)):
    #     print(i)
    #     item = random.randrange(0, len(train_dataset))
    #     item = item
    #     img, boxes, labels = train_dataset.testGet(item)
    #     # img, boxes = train_dataset[item]
    #     img = img.numpy().transpose(1, 2, 0).copy()*255
    #     train_dataset.data_encoder.test_encode(boxes, img, labels)
    #
    #     boxes = boxes.numpy().tolist()
    #     w, h, _ = img.shape
    #     # print('img', img.shape)
    #     # print('boxes', boxes.shape)
    #
    #     for box in boxes:
    #         x1 = int(box[0]*w)
    #         y1 = int(box[1]*h)
    #         x2 = int(box[2]*w)
    #         y2 = int(box[3]*h)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    #         boxw = x2-x1
    #         boxh = y2-y1
    #         print(boxw,boxh, box)
    #         if boxw is 0 or boxh is 0:
    #             raise 'zero width'
    #
    #     cv2.imwrite('test'+str(i)+'.jpg', img)
    #     if i == 0:
    #         break