# encoding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLayer(nn.Module):
    num_classes = 3                                 # 3类，是车牌或车辆或背景
    num_anchors = [21, 1, 1]                        # 21=4*4+2*2+1（对应anchor的大小32,64,128，保证anchor密度相同）
    in_planes = [128, 256, 256]                     # Inception3通道数, Conv3_2通道数, Conv4_2通道数

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList()           # 预测坐标
        self.conf_layers = nn.ModuleList()          # 预测置信度

        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 3, kernel_size=3, padding=1))

    def forward(self, xs):
        '''
        xs:list of 之前的feature map list
        xs含有三个矩阵，size分别是： [(-1, 128, 32, 32), (-1, 256, 16, 16), (-1, 256, 8, 8)]
        retrun: loc_preds: [N, 21842, 4]
                conf_preds:[N, 21842, 2]
        '''

        y_locs = []
        y_confs = []

        # 循环处理每一层的矩阵
        for i, x in enumerate(xs):
            # 处理坐标
            y_loc = self.loc_layers[i](x)
            # print('loc MultiBoxLayer_Conv2d', y_loc.size())       # (-1, 84, 32, 32) (-1, 4, 16, 16) (-1, 4, 8, 8)
            N = y_loc.size(0)                                       # N为维度，类似通道数c
            # print('N', N)                                         # (-1)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()          # permute 置换|转置
            # print('loc MultiBoxLayer_permute', y_loc.size())      # (-1, 32, 32, 84) (-1, 16, 16, 4) (-1, 8, 8, 4)
            y_loc = y_loc.view(N, -1, 4)                            # 减少一个维度
            # print('loc MultiBoxLayer_view', y_loc.size())         # (-1, 21504, 4)   (-1, 256, 4)    (-1, 64, 4)
            y_locs.append(y_loc)

            # 处理置信度
            y_conf = self.conf_layers[i](x)
            # print('y_conf MultiBoxLayer_y_conf', y_conf.size())   # (-1, 63, 32, 32) (-1, 3, 16, 16)  (-1, 3, 8, 8)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            # print('y_conf MultiBoxLayer_permute', y_conf.size())  # (-1, 32, 32, 63) (-1, 16, 16, 3)  (-1, 8, 8, 4)
            y_conf = y_conf.view(N, -1, 3)
            # print('y_conf MultiBoxLayer_view', y_conf.size())     # (-1, 21504, 3)   (-1, 256, 3)     (-1, 64, 3)
            y_confs.append(y_conf)

            # print('y_locs', len(y_locs))
            # print('y_confs', len(y_confs))

        loc_preds = torch.cat(y_locs, 1)                            # (-1, 21824, 4)   21504+256+64=21824
        conf_preds = torch.cat(y_confs, 1)                          # (-1, 21824, 3)   21504+256+64=21824
        return loc_preds, conf_preds


def multi_box_layer_test():
    hs = []
    data = torch.randn(1, 128, 32, 32)
    hs.append(data)
    data = torch.randn(1, 256, 16, 16)
    hs.append(data)
    data = torch.randn(1, 256, 8, 8)
    hs.append(data)

    multilbox = MultiBoxLayer()

    loc_preds, conf_preds = multilbox(hs)
    print('output: ', loc_preds.size(), conf_preds.size())


if __name__ == '__main__':
    multi_box_layer_test()
