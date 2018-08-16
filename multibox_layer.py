# encoding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLayer(nn.Module):
    num_classes = 2                                 # 2类，是人脸或是背景
    num_anchors = [21, 1, 1]
    in_planes = [128, 256, 256]                     # Inception3, Conv3_2, Conv4_2

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*2, kernel_size=3, padding=1))

    def forward(self, xs):
        '''
        xs:list of 之前的feature map list
        retrun: loc_preds: [N, 21842, 4]
                conf_preds:[N, 21842, 2]
        '''

        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)                           # N,anhors*4,H,W
            # print('MultiBoxLayer_Conv2d', y_loc.size())
            N = y_loc.size(0)                                       # N分别为：84，4，4
            # print('N', N)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()          # permute 置换|转置
            # print('MultiBoxLayer_permute', y_loc.size())
            y_loc = y_loc.view(N, -1, 4)
            # print('MultiBoxLayer_view', y_loc.size())
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            # print('MultiBoxLayer_y_conf', y_conf.size())
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            # print('MultiBoxLayer_permute', y_conf.size())
            y_conf = y_conf.view(N, -1, 2)
            # print('MultiBoxLayer_view', y_conf.size())
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds