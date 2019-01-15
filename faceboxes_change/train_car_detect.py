# encoding:utf-8
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from networks_multi import FaceBox
from multibox_loss_multi import MultiBoxLoss
from dataset import ListDataset
import common
import cv2
import numpy as np
from encoderl import DataEncoder
import time
from argparse import ArgumentParser


def show_img(img, boxes):
    print(img.size())
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2RGBA)
    h, w, c = img.shape
    print(h, w, c)
    for i in range(len(boxes)):
        box = boxes[i]
        print('box', box)
        # cv2.circle(img, (int(output[2 * i]), int(output[2 * i + 1])), 3, (0, 0, 255), -1)
        cv2.rectangle(img, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0, 255, 0), 2)
    cv2.imshow('show_img', img)
    cv2.waitKey(0)


class ModuleTrain:
    def __init__(self, train_path, test_path, model_file, model, img_size=1024, batch_size=16, lr=1e-3,
                 re_train=False, best_loss=2, use_gpu=False, nms_threshold=0.5):
        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.re_train = re_train                        # 不加载训练模型，重新进行训练
        self.best_loss = best_loss                      # 最好的损失值，小于这个值，才会保存模型
        self.use_gpu = False
        self.nms_threshold = nms_threshold

        if use_gpu is True:
            print("gpu available: %s" % str(torch.cuda.is_available()))
            if torch.cuda.is_available():
                self.use_gpu = True
            else:
                self.use_gpu = False

        # 模型
        self.model = model

        if self.use_gpu:
            print('[use gpu] ...')
            self.model = self.model.cuda()

        # 加载模型
        if os.path.exists(self.model_file) and not self.re_train:
            self.load(self.model_file)

        # RandomHorizontalFlip
        self.transform_train = T.Compose([
            # T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

        self.transform_test = T.Compose([
            # T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # Dataset
        train_label = os.path.join(self.train_path, 'faceboxes_label.txt')
        test_label = os.path.join(self.test_path, 'faceboxes_label.txt')
        train_dataset = ListDataset(root=self.train_path, list_file=train_label, train=True, transform=self.transform_train)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_dataset = ListDataset(root=self.test_path, list_file=test_label, train=False, transform=self.transform_test)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        print('train_loader len: %d' % len(self.train_loader.dataset))
        print(' test_loader len: %d' % len(self.test_loader.dataset))

        self.criterion = MultiBoxLoss()

        self.lr = lr
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

        pass

    def train(self, epoch, decay_epoch=60, save_best=True):

        for epoch_i in range(epoch):
            self.model.train()

            train_loss = 0.0
            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:
                self.lr *= 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            # if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
            #     self.lr = self.lr * 0.1
            #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

            print('================================================')
            for i, (images, loc_targets, conf_targets) in enumerate(self.train_loader):
                images = Variable(images)
                loc_targets = Variable(loc_targets)
                conf_targets = Variable(conf_targets)
                if self.use_gpu:
                    images, loc_targets, conf_targets = images.cuda(), loc_targets.cuda(), conf_targets.cuda()

                loc_preds, conf_preds = self.model(images)
                loss = self.criterion(loc_preds, loc_targets, conf_preds, conf_targets)  # 计算损失，用MultiBoxLoss
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss /= len(self.train_loader)
            print ('[Train] Epoch [%d/%d] average_loss: %.6f lr: %.6f' % (epoch_i + 1, epoch, train_loss, self.lr))

            test_loss = self.test()
            if save_best is True:
                if self.best_loss > test_loss:
                    self.best_loss = test_loss
                    str_list = self.model_file.split('.')
                    best_model_file = ""
                    for str_index in range(len(str_list)):
                        best_model_file = best_model_file + str_list[str_index]
                        if str_index == (len(str_list) - 2):
                            best_model_file += '_best'
                        if str_index != (len(str_list) - 1):
                            best_model_file += '.'
                    self.save(best_model_file)                                  # 保存最好的模型

        self.save(self.model_file)

    def test(self, show_info=False):
        self.model.eval()
        test_loss = 0
        data_encoder = DataEncoder()

        time_start = time.time()
        # 测试集
        # for data, target in test_loader:
        for images, loc_targets, conf_targets in self.test_loader:
            images, loc_targets, conf_targets = Variable(images), Variable(loc_targets), Variable(conf_targets)

            if self.use_gpu:
                images, loc_targets, conf_targets = images.cuda(), loc_targets.cuda(), conf_targets.cuda()

            # print('images', images.size())
            loc_preds, conf_preds = self.model(images)
            loss = self.criterion(loc_preds, loc_targets, conf_preds, conf_targets)  # 计算损失
            test_loss += loss.item()

            if show_info is True:
                # print('0 pre_label', loc_preds.size(), conf_preds.size())
                # print('0 pre_label', loc_preds, conf_preds)

                print("loc_preds len: ", len(loc_preds))
                for i in range(len(loc_preds)):
                    # print('2 pre_label', loc_preds[i].size(), conf_preds[i].size())
                    # print('3 pre_label', loc_preds[i], conf_preds[i])
                    boxes, labels, max_conf = data_encoder.decode(loc_preds[i], conf_preds[i], self.use_gpu, self.nms_threshold)
                    # print('boxes', boxes)
                    # print('labels', labels)
                    # print('max_conf', max_conf)

                    show_img(images[i].permute(1, 2, 0), boxes.cpu().detach().numpy())
                    # show_img(images[i], boxes.cpu().detach().numpy())

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))
        avg_loss = test_loss / len(self.test_loader)
        print('[Test] avg_loss: {:.6f} time: {:.6f}\n'.format(avg_loss, time_avg))
        return avg_loss

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)


def parse_argvs():
    parser = ArgumentParser(description='car_classifier')
    parser.add_argument('--train_path', type=str, help='train dataset path',
                        default='../../Data/yolo/yolo_data_new/car_detect_train/')
    parser.add_argument('--test_path', type=str, help='test dataset path',
                        default='../../Data/yolo/yolo_data_new/car_detect_train/')

    parser.add_argument("--output_model_path", type=str, help="output model path", default='./weight/car_rough_detect.pt')
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--img_size', type=int, help='img size', default=1024)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--nms_threshold', type=float, help='learning rate', default=0.0)
    parser.add_argument('--cuda', type=bool, help='use gpu', default=True)

    input_args = parser.parse_args()
    print(input_args)
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    # train_path = os.path.expanduser('~/deeplearning/Data/car_rough_detect/car_detect_train/')
    # test_path = os.path.expanduser('~/deeplearning/Data/car_rough_detect/car_detect_test/')

    # common.mkdir_if_not_exist('./weight')
    # output_model_path = 'weight/car_rough_detect.pt'

    # batch_size = 20
    # lr = 0.001
    # img_size = 1024

    model = FaceBox()
    model_train = ModuleTrain(train_path=args.train_path, test_path=args.test_path, model_file=args.output_model_path,
                              model=model, batch_size=args.batch_size, img_size=args.img_size,
                              lr=args.lr, use_gpu=args.cuda, nms_threshold=args.nms_threshold)

    model_train.train(100, 60)
    # model_train.test(show_info=True)

