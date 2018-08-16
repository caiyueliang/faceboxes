# encoding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from networks import FaceBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset
from common import common
import visdom
import numpy as np


use_gpu = torch.cuda.is_available()

re_train = False
learning_rate = 0.001
num_epochs = 200
decay_epoch = 60
batch_size = 16


if __name__ == '__main__':
    train_root = os.path.expanduser('~/deeplearning/Data/car_rough_detect/car_detect_train/')
    train_label = './label/car_detect_train_label.txt'
    test_root = os.path.expanduser('~/deeplearning/Data/car_rough_detect/car_detect_test/')
    test_label = './label/car_detect_test_label.txt'

    common.mkdir_if_not_exist('./weight')
    model_file = 'weight/car_detect.pt'                 # 保存的模型名称

    best_loss = 10

    net = FaceBox()
    if use_gpu:
        net.cuda()

    # 加载模型
    if os.path.exists(model_file) and not re_train:
        print('[load model] ...', model_file)
        net.load_state_dict(torch.load(model_file))

    criterion = MultiBoxLoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0003)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_dataset = ListDataset(root=train_root, list_file=train_label, train=True,
                                transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % batch_size)

    # 可视化工具
    num_iter = 0
    vis = visdom.Visdom()
    win = vis.line(Y=np.array([0]), X=np.array([0]))

    net.train()
    for epoch in range(num_epochs):
        if epoch >= decay_epoch and epoch % decay_epoch == 0:
            learning_rate *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\nEpoch [%d/%d] learning_rate:%f' % (epoch + 1, num_epochs, learning_rate))
        total_loss = 0.

        for i, (images, loc_targets, conf_targets) in enumerate(train_loader):
            images = Variable(images)
            loc_targets = Variable(loc_targets)
            conf_targets = Variable(conf_targets)
            if use_gpu:
                images, loc_targets, conf_targets = images.cuda(), loc_targets.cuda(), conf_targets.cuda()

            loc_preds, conf_preds = net(images)
            loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)          # 计算损失
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [%d/%d] average_loss: %.4f' % (epoch + 1, num_epochs, total_loss / len(train_loader)))
        num_iter = num_iter + 1
        vis.line(Y=np.array([total_loss / len(train_loader)]), X=np.array([num_iter]), win=win, update='append')

        # 保存最好的模型
        if best_loss > (total_loss / len(train_loader)):
            best_loss = total_loss / len(train_loader)
            str_list = model_file.split('.')
            best_model_file = ""
            for str_index in range(len(str_list)):
                best_model_file = best_model_file + str_list[str_index]
                if str_index == (len(str_list) - 2):
                    best_model_file += '_best'
                if str_index != (len(str_list) - 1):
                    best_model_file += '.'
                print('[saving best model] ...', best_model_file)
                torch.save(net.state_dict(), best_model_file)

    print('[saving model] ...', model_file)
    torch.save(net.state_dict(), model_file)

