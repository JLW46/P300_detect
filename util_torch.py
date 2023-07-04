import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import math
from math import cos, pi
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class weightConstraint(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("Entered")
            w = module.weight.data
            w = w.clamp(self.min, self.max)
            module.weight.data = w


class EEGNET(nn.Module):
    # len=125
    def __init__(self, eeg_ch):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(EEGNET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(1, 63), stride=(1, 1),
                               padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv_spatial = nn.Conv2d(in_channels=8, out_channels=16,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(1, 13), stride=(1, 1),
                               padding='same', groups=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding='valid')
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(80, 1)

    def forward(self, x):
        # Block 1
        x = self.bn1(self.conv1(x))  # [8, ch, 125]
        x = self.bn2(self.conv_spatial(x))  # [16, 1, 125]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 5))), p=0.25)  # [16, 1, 25]
        # Block 2
        x = self.conv3(x)
        x = self.bn3(self.conv4(x))  # [16, 1, 25]
        x = func.dropout(input=(func.avg_pool2d(func.elu(x), (1, 5))), p=0.5)  # [16, 1, 5]
        x = torch.flatten(input=x, start_dim=1)  # [80]
        x = func.sigmoid(self.fc1(x))  # [2]
        return x


class EEGNET_Res_B(nn.Module):
    # len=125
    def __init__(self, eeg_ch):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(EEGNET_Res_B, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(1, 63), stride=(1, 1),
                               padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv_spatial = nn.Conv2d(in_channels=8, out_channels=16,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv_res_1 = nn.Conv2d(in_channels=16, out_channels=8,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding='same')
        self.conv_res_21 = nn.Conv2d(in_channels=16, out_channels=8,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     padding='same')
        self.conv_res_22 = nn.Conv2d(in_channels=8, out_channels=8,
                                     kernel_size=(1, 13), stride=(1, 1),
                                     padding='same')

        self.bn_res = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(80, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = self.bn1(self.conv1(x))  # [8, ch, 125]
        x = self.bn2(self.conv_spatial(x))  # [16, 1, 125]
        x = self.dropout1(func.avg_pool2d(func.elu(x), (1, 5)))  # [16, 1, 25]
        # Block 2

        # x_1 = func.relu(self.bn_21(self.conv_21(x)))
        x_1 = self.conv_res_1(x)
        x_2 = self.conv_res_21(x)
        x_2 = self.conv_res_22(x_2)
        x_res = torch.cat((x_1, x_2), 1)
        x = func.relu(self.bn_res(x + x_res))

        x = self.dropout2(func.avg_pool2d(x, (1, 5)))  # [16, 1, 5]
        x = torch.flatten(input=x, start_dim=1)

        x = func.sigmoid(self.fc1(x))  # [2]
        return x


class EEGNET_Res_C(nn.Module):
    # len=125 , on RES0, replace final avg_pool with conv reduct and add 1 more res
    def __init__(self, eeg_ch, num_res_module_1=1, num_res_module_2=1):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(EEGNET_Res_C, self).__init__()
        self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=8,
                                       kernel_size=(1, 63), stride=(1, 1),
                                       padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8)
        spatial_out = 16
        self.conv_spatial = nn.Conv2d(in_channels=8, out_channels=spatial_out,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=8)
        self.bn2 = nn.BatchNorm2d(num_features=spatial_out)
        self.dropout1 = nn.Dropout(p=0.25)
        self.res_module_1 = nn.ModuleList([])
        num_res1_branch = 2
        for i in range(num_res_module_1):
            self.res_module_1.append(nn.ModuleList([
                nn.Conv2d(in_channels=spatial_out, out_channels=spatial_out // num_res1_branch,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 1
                nn.Conv2d(in_channels=spatial_out, out_channels=spatial_out // num_res1_branch,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=spatial_out // num_res1_branch, out_channels=spatial_out // num_res1_branch,
                          kernel_size=(1, 13), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.BatchNorm2d(num_features=spatial_out)
            ]))
        reduct2_out = 2 * spatial_out
        self.conv_reduction21 = nn.Conv2d(in_channels=spatial_out, out_channels=reduct2_out,
                                          kernel_size=(1, 3), stride=(1, 2),
                                          padding='valid', groups=spatial_out)
        self.conv_reduction22 = nn.Conv2d(in_channels=reduct2_out, out_channels=reduct2_out,
                                          kernel_size=(1, 1), stride=(1, 1),
                                          padding='same')
        self.bn_red2 = nn.BatchNorm2d(num_features=reduct2_out)
        self.res_module_2 = nn.ModuleList([])
        num_res2_branch = 2
        for i in range(num_res_module_2):
            self.res_module_2.append(nn.ModuleList([
                nn.Conv2d(in_channels=reduct2_out, out_channels=reduct2_out // num_res2_branch,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 1
                nn.Conv2d(in_channels=reduct2_out, out_channels=reduct2_out // num_res2_branch,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=reduct2_out // num_res2_branch, out_channels=reduct2_out // num_res2_branch,
                          kernel_size=(1, 3), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.BatchNorm2d(num_features=reduct2_out)
            ]))
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2 * 96, 1)

    def forward(self, x):
        b = x.size()[0]
        # Block 1
        x = self.bn1(self.conv_temporal(x))  # [8, ch, 125]
        x = self.bn2(self.conv_spatial(x))  # [16, 1, 125]
        x = func.avg_pool2d(func.elu(x), (1, 5))  # [16, 1, 25]
        x = self.dropout1(x)
        # Block res1
        for conv_res1_1, conv_res1_21, conv_res1_22, bn in self.res_module_1:
            x_re1_1 = conv_res1_1(x)
            x_re1_2 = conv_res1_22(conv_res1_21(x))
            x = bn(x + torch.cat((x_re1_1, x_re1_2), dim=1))
            x = func.relu(x)  # [16, 1, 25]
        # Reduction
        x = self.conv_reduction22(self.conv_reduction21(x))  # [16, 1, 12]
        x = func.relu(self.bn_red2(x))
        # Block res2
        for conv_res1_1, conv_res1_21, conv_res1_22, bn in self.res_module_2:
            x_re1_1 = conv_res1_1(x)
            x_re1_2 = conv_res1_22(conv_res1_21(x))
            x = bn(x + torch.cat((x_re1_1, x_re1_2), dim=1))
            x = func.relu(x)
        # Out
        x = func.avg_pool2d(x, (1, 2))  # [32, 1, 6]
        x = torch.flatten(input=x, start_dim=1)
        x = self.dropout2(x)
        out = func.sigmoid(self.fc1(x))
        return out


class EEGNET_Res_D(nn.Module):
    # len=125
    def __init__(self, eeg_ch):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(EEGNET_Res_D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(1, 63), stride=(1, 1),
                               padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv_spatial = nn.Conv2d(in_channels=8, out_channels=16,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=8)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv_res_1 = nn.Conv2d(in_channels=16, out_channels=32,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding='same')
        self.conv_res_21 = nn.Conv2d(in_channels=32, out_channels=32,
                                     kernel_size=(1, 5), stride=(1, 1),
                                     padding='same')
        self.conv_res_22 = nn.Conv2d(in_channels=32, out_channels=32,
                                     kernel_size=(1, 13), stride=(1, 1),
                                     padding='same')
        self.conv_res_3 = nn.Conv2d(in_channels=64, out_channels=16,
                                    kernel_size=(1, 1), stride=(1, 1),
                                    padding='same')
        self.bn_res = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = self.bn1(self.conv1(x))  # [8, ch, 125]
        x = self.bn2(self.conv_spatial(x))  # [16, 1, 125]
        x = self.dropout1(func.avg_pool2d(func.elu(x), (1, 5)))  # [16, 1, 25]
        # Block 2

        x_1 = self.conv_res_1(x)
        x_21 = self.conv_res_21(x_1)  # branch 1
        x_22 = self.conv_res_22(x_1)  # branch 2
        x_2 = torch.cat((x_21, x_22), dim=1)
        x_3 = self.conv_res_3(x_2)
        x = func.relu(self.bn_res(x + x_3))

        x = self.dropout2(func.avg_pool2d(x, (1, 3)))  # [16, 1, 5]
        x = torch.flatten(input=x, start_dim=1)
        x = func.sigmoid(self.fc1(x))  # [2]
        return x


class RESNET(nn.Module):
    # len=125
    def __init__(self, eeg_ch, num_res_module_1, num_reduct_module_1):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(RESNET, self).__init__()

        self.stem_conv11 = nn.Conv2d(in_channels=1, out_channels=4,
                                     kernel_size=(1, 5), stride=(1, 1),
                                     padding='same')
        self.stem_conv12 = nn.Conv2d(in_channels=1, out_channels=4,
                                     kernel_size=(1, 15), stride=(1, 1),
                                     padding='same')
        self.stem_conv13 = nn.Conv2d(in_channels=1, out_channels=4,
                                     kernel_size=(1, 25), stride=(1, 1),
                                     padding='same')
        self.stem_conv21 = nn.Conv2d(in_channels=12, out_channels=48,
                                     kernel_size=(1, 3), stride=(1, 2),
                                     padding='valid', groups=12)
        self.res_module_1 = nn.ModuleList([])
        for i in range(num_res_module_1):
            self.res_module_1.append(nn.ModuleList([
                nn.Conv2d(in_channels=48, out_channels=8,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 1
                nn.Conv2d(in_channels=48, out_channels=8,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=8, out_channels=8,
                          kernel_size=(1, 5), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=8, out_channels=8,
                          kernel_size=(eeg_ch, 1), stride=(1, 1),
                          padding='same', padding_mode='circular'),  # branch 2
                nn.Conv2d(in_channels=16, out_channels=48,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch merged
            ]))
        self.reduct_conv1 = nn.Conv2d(in_channels=48, out_channels=96,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=48)
        self.reduct_module_1 = nn.ModuleList([])
        for i in range(num_reduct_module_1):
            self.reduct_module_1.append(nn.ModuleList([
                nn.Conv2d(in_channels=96, out_channels=32,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 1 (+ave_pool)
                nn.Conv2d(in_channels=96, out_channels=32,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=32, out_channels=32,
                          kernel_size=(1, 3), stride=(1, 2),
                          padding='valid'),  # branch 2
                nn.Conv2d(in_channels=96, out_channels=32,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 3
                nn.Conv2d(in_channels=32, out_channels=32,
                          kernel_size=(1, 3), stride=(1, 2),
                          padding='valid'),  # branch 3
                nn.Conv2d(in_channels=32, out_channels=32,
                          kernel_size=(1, 3), stride=(1, 1),
                          padding='same'),  # branch 3
            ]))
        self.fc1 = nn.Linear(384, 2)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv_end = nn.Conv2d(in_channels=1, out_channels=4,
                                  kernel_size=(1, 25), stride=(1, 1),
                                  padding='same')

    def forward(self, x):
        # [1, 64, 125]
        x = torch.cat((self.stem_conv11(x),
                       self.stem_conv12(x),
                       self.stem_conv13(x)), dim=1)
        # [12, 64, 125]
        x = self.stem_conv21(x)
        # [48, 64, 62]
        x = func.max_pool2d(func.relu(x), (1, 3))
        # [48, 64, 20]
        # x = torch.cat((self.stem_conv21(x),
        #                func.max_pool2d(func.elu(x), (1, 3), stride=(1, 2))))
        for res_conv_11, res_conv_21, res_conv_22, res_conv_23, res_conv_m in self.res_module_1:
            x_1 = res_conv_11(x)  # 8,64,20
            x_21 = res_conv_21(x)  # 8,64,20
            x_2 = res_conv_22(x_21)  # 8,64,20
            x_2 = res_conv_23(x_2)  # 8,64,20
            x_m = res_conv_m(torch.cat((x_1, x_2), dim=1))  # 48,64,20
            x = func.relu(x + x_m)
        x = self.reduct_conv1(x)
        x = self.drop1(x)
        # [96, 1, 20]
        for red_conv_11, red_conv_21, red_conv_22, red_conv_31, red_conv_32, red_conv_33 in self.reduct_module_1:
            x_1 = red_conv_11(func.avg_pool2d(x, (1, 3), stride=(1, 2)))
            x_2 = red_conv_21(x)  # 32,1,20
            x_2 = red_conv_22(x_2)  # 32,1,9
            x_31 = red_conv_31(x)  # 32,1,20
            x_32 = red_conv_32(x_31)  # 32,1,9
            x_33 = red_conv_33(x_32)  # 32,1,9
            x = torch.cat((x_1, x_2, x_33), dim=1)
        # [96, 1, 9]

        x = func.avg_pool2d(x, (1, 3), stride=(1, 2))
        x = torch.flatten(input=x, start_dim=1)
        # x = func.dropout(x, p=0.5)
        x = self.drop2(x)
        x = self.fc1(x)
        out = func.softmax(x, dim=-1)

        return out


class RESNET_s(nn.Module):
    # len=125
    def __init__(self, eeg_ch, num_res_module_1, num_reduct_module_1):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(RESNET_s, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=(64, 1), stride=(1, 1),
                               padding='same', padding_mode='circular')
        self.stem_conv11 = nn.Conv2d(in_channels=4, out_channels=8,
                                     kernel_size=(1, 5), stride=(1, 1),
                                     padding='same')
        self.stem_conv12 = nn.Conv2d(in_channels=4, out_channels=8,
                                     kernel_size=(1, 25), stride=(1, 1),
                                     padding='same')
        self.stem_conv13 = nn.Conv2d(in_channels=4, out_channels=8,
                                     kernel_size=(1, 50), stride=(1, 1),
                                     padding='same')
        self.stem_conv21 = nn.Conv2d(in_channels=24, out_channels=48,
                                     kernel_size=(1, 3), stride=(1, 2),
                                     padding='valid', groups=12)
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(96)
        self.res_module_1 = nn.ModuleList([])
        for i in range(num_res_module_1):
            self.res_module_1.append(nn.ModuleList([
                nn.Conv2d(in_channels=48, out_channels=8,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 1
                nn.Conv2d(in_channels=48, out_channels=8,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=8, out_channels=8,
                          kernel_size=(1, 5), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=8, out_channels=8,
                          kernel_size=(1, 5), stride=(1, 1),
                          padding='same'),  # branch 2
                nn.Conv2d(in_channels=16, out_channels=48,
                          kernel_size=(1, 1), stride=(1, 1),
                          padding='same'),  # branch merged
                nn.BatchNorm2d(48)
            ]))
        self.reduct_conv1 = nn.Conv2d(in_channels=48, out_channels=48,
                                      kernel_size=(eeg_ch, 1), stride=(1, 1),
                                      padding='valid', groups=48)
        self.fc1 = nn.Linear(432, 2)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.conv_end = nn.Conv2d(in_channels=1, out_channels=4,
                                  kernel_size=(1, 25), stride=(1, 1),
                                  padding='same')

    def forward(self, x):
        # [1, 64, 125]
        x = self.conv1(x)
        x = torch.cat((self.stem_conv11(x),
                       self.stem_conv12(x),
                       self.stem_conv13(x)), dim=1)
        # [12, 64, 125]
        x = self.stem_conv21(x)
        x = self.reduct_conv1(x)  # 48 1 62
        x = self.bn1(x)
        # [48, 1, 62]
        x = func.max_pool2d(func.relu(x), (1, 3))
        # [48, 1, 20]
        for res_conv_11, res_conv_21, res_conv_22, res_conv_23, res_conv_m, bn in self.res_module_1:
            x_1 = res_conv_11(x)  # 8,1,20
            x_21 = res_conv_21(x)  # 8,1,20
            x_2 = res_conv_22(x_21)  # 8,1,20
            x_2 = res_conv_23(x_2)  # 8,1,20
            x_m = res_conv_m(torch.cat((x_1, x_2), dim=1))  # 48,1,20
            x = func.relu(x + x_m)

        x = func.avg_pool2d(x, (1, 3), stride=(1, 2))
        x = torch.flatten(input=x, start_dim=1)

        x = self.drop2(x)
        x = self.fc1(x)
        out = func.softmax(x, dim=-1)

        return out


class VIT(nn.Module):
    def __init__(self, num_eegch, num_heads, num_layers):
        # in_shape = [C_ch, H_eegch, W_time] [1, 64, 125]
        super(VIT, self).__init__()
        self.num_projected_features = 64
        self.qkv_len = 16
        self.scale = self.qkv_len ** (-0.5)
        self.h = num_eegch
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.p = 5
        self.projection = nn.Linear(self.h * 25, self.num_projected_features)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.num_projected_features))
        self.pe = nn.Parameter(torch.randn(1, self.p + 1, 1))

        self.transformer_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            self.transformer_layers.append(nn.ModuleList([
                nn.Linear(self.num_projected_features, 3 * self.qkv_len * self.num_heads),  # make qkv
                nn.Softmax(dim=-1),  # scaled dot product att softmax
                nn.Linear(self.num_heads * self.qkv_len, self.num_projected_features),  # multi head out
                nn.LayerNorm(self.num_projected_features),
                nn.LayerNorm(self.num_projected_features),
                nn.Sequential(nn.Linear(self.num_projected_features, 2 * self.num_projected_features),
                              nn.GELU(),
                              nn.Dropout(0.25),
                              nn.Linear(2 * self.num_projected_features, self.num_projected_features),
                              nn.Dropout(0.25))
            ]))
        self.MLP_head = nn.Sequential(nn.LayerNorm(self.num_projected_features),
                                      nn.Dropout(0.5),
                                      nn.Linear(self.num_projected_features, 2),
                                      nn.Softmax(dim=-1))
        # self.make_qkv = nn.Linear(self.num_projected_features, 3 * self.qkv_len * self.num_heads),
        # self.softmax = nn.Softmax(dim=-1),
        # self.multi_head_linear = nn.Linear(self.num_heads * self.qkv_len, self.num_projected_features)

    def forward(self, x):
        b = x.size()[0]
        # Patching and positional embedding x[b, c=1, h=num_ch, w=125] --> [b, p + 1, f]
        x = torch.transpose(torch.reshape(x, (-1, 1, self.h, self.p, 25)), 3, 4)
        # x[b, c=1, h=num_ch, w=125] --> [b, c=1, h=num_ch, p=5, f=25] --> [b, c=1, h=num_ch, f=25, p=5]
        x = torch.transpose(torch.reshape(x, (-1, self.h * 25, self.p)), 1, 2)  # [b, p, h*25]
        # x[b, c=1, h=num_ch, f=25, p=5] --> [b, h*f=num_ch*25, p=5] --> [b, p=5, h*f=num_ch*25]
        x = self.projection(x)
        # x[b, p=5, h*f=num_ch*25] --> [b, p=5, proj_f=64]
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        # x[b, p=5, proj_f=64] cat token[b, 1, proj_f=64] --> [b, 5+1, proj_f]
        x = x + self.pe.repeat(1, 1, self.num_projected_features)
        # transformer
        for make_qkv, softmax, multi_head_out, layer_norm_1, layer_norm_2, MLP in self.transformer_layers:
            ##### QKV #####
            qkv = make_qkv(layer_norm_1(x)).chunk(3, dim=-1)
            # x[b, p+1, proj_f] --> [b, p+1, qkv_len*num_heads]*3
            q = qkv[0].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2)
            # q: [b, p+1, qkv_len*num_heads] --> [b, p+1, num_heads, qkv_len] --> [b, num_heads, p+1, qkv_len]
            k = qkv[1].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2).transpose(2, 3)
            # k: [b, p+1, qkv_len*num_heads] --> [b, p+1, num_heads, qkv_len] --> [b, num_heads, p+1, qkv_len] -> [b, num_heads, qkv_len, p+1]
            v = qkv[2].reshape((b, self.p + 1, self.num_heads, self.qkv_len)).transpose(1, 2)
            # v: [b, p+1, qkv_len*num_heads] --> [b, p+1, num_heads, qkv_len] --> [b, num_heads, p+1, qkv_len]
            ##### Scaled Dot-Product Att #####
            # [b, num_heads, p+1, qkv_len] X [b, num_heads, qkv_len, p+1] X [b, num_heads, p+1, qkv_len] -> [b, num_heads, p+1, qkv_len]
            scaled_dot_produc_att = torch.matmul(softmax(torch.matmul(q, k) * self.scale), v)
            # [b, num_heads, p+1, qkv_len] -> [b, p+1, num_heads*qkv_len] -> [b, p+1, projected_len]
            multi_head_att = multi_head_out(
                scaled_dot_produc_att.transpose(1, 2).reshape((b, self.p + 1, self.num_heads * self.qkv_len)))
            x = multi_head_att + x
            ##### MLP #####
            x = MLP(layer_norm_2(x)) + x
        out = self.MLP_head(x[:, 0, :])
        return out


# a = torch.randn(10, 3, 2, 4)
# print(a[0, 1, 1, :])
# a = a.transpose(1, 2) # [10, 2, 3, 4]
# print(a[0, 1, 1, :])
# a = a.reshape(10, 2, 12)
# print(a[0, 1, :])


class EegData(torch.utils.data.Dataset):
    def __init__(self, raw_x, raw_y, class_weight):
        """
        raw: raw eeg samples in numpy [n_batch, c, ch_eeg, time]
        """
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.class_weight = class_weight

    def __len__(self):
        return np.shape(self.raw_x)[0]

    def __getitem__(self, item):
        sample_weight = []
        if torch.is_tensor(item):
            item = item.tolist()
        if torch.is_tensor(self.raw_x):
            if self.raw_y[item] == 1:
                weight = self.class_weight[0]
            else:
                weight = self.class_weight[1]
            sample = {'x': self.raw_x[item, :, :, :],
                      'y': self.raw_y[item],
                      'weight': weight}
        else:
            if self.raw_y[item] == 1:
                weight = self.class_weight[0]
            else:
                weight = self.class_weight[1]
            sample = {'x': torch.from_numpy(self.raw_x[item, :, :, :]).float(),
                      'y': torch.from_numpy(np.array(self.raw_y[item])).float(),
                      'weight': torch.from_numpy(np.array(weight)).float()}

        return sample


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _manual_val_split(X, Y, ratio):
    pos_ind = np.where(Y[:] == 1)[0]
    neg_ind = np.where(Y[:] == 0)[0]
    random.shuffle(pos_ind)
    random.shuffle(neg_ind)
    pos_train = pos_ind[:int(len(pos_ind) * ratio)]
    pos_val = pos_ind[int(len(pos_ind) * ratio):]
    neg_train = neg_ind[:int(len(neg_ind) * ratio)]
    neg_val = neg_ind[int(len(neg_ind) * ratio):]
    X_train = np.concatenate([X[pos_train], X[neg_train]], axis=0)
    X_val = np.concatenate([X[pos_val], X[neg_val]], axis=0)
    Y_train = np.concatenate([Y[pos_train], Y[neg_train]], axis=0)
    Y_val = np.concatenate([Y[pos_val], Y[neg_val]], axis=0)
    return X_train, Y_train, X_val, Y_val


def draw_roc(fpr, tpr, roc_auc, stage):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(stage)
    plt.legend(loc="lower right")
    # plt.savefig('roc.png', )
    plt.show()

def draw_acc(ba_acc, thresholds):
    plt.figure()
    lw = 2
    plt.plot(thresholds, ba_acc, color='darkorange', lw=lw,
             label='ACC-Threshold_curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Thresholds')
    plt.ylabel('Balanced_ACC')
    plt.legend(loc="lower right")
    # plt.savefig('roc.png', )
    plt.show()


def draw_pred(preds, true):
    preds_p = []
    preds_n = []
    for i in range(len(preds)):
        preds[i] = preds[i].cpu().detach().numpy()
        true[i] = true[i].cpu().detach().numpy()
        pos_ind = np.where((true[i] == 1))
        preds_p.append(preds[i][pos_ind])
        preds_n.append(np.delete(preds[i], pos_ind))


    distr = []
    distr.append(preds_p)
    distr.append(preds_n)

    label_place = int(0.5 * 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(str('Distribution'), fontsize=14)
    # ax1.set_ylabel('Balanced Accuracy', fontsize=24)
    ax1.set_ylabel('Prediction', fontsize=14)
    ax1.set_xlabel('Stage of training', fontsize=14)
    COLOR = ['mistyrose', 'mediumpurple', 'powderblue', 'gold', 'red', 'blue', 'rosybrown', 'red', 'blue']
    boxplots = []
    legends = []
    for i in range(2):

        labels = ['train', 'val', 'test', 'testext']
        x_spacing = np.array(range(4)) * (2 + label_place + 2) + i
        bp = ax1.boxplot(distr[i], labels=labels, positions=x_spacing, patch_artist=True, showmeans=True,
                         boxprops=dict(facecolor=COLOR[i], linewidth=2),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2),
                         flierprops=dict(marker='+', markersize=12),
                         meanprops=dict(markersize=12, color='black'),
                         medianprops=dict(linewidth=3, color='black'))
        boxplots.append(bp)
        legends.append(bp["boxes"][0])
    font = font_manager.FontProperties(style='normal', size=14)
    ax1.legend(legends, ['target', 'non_target'], loc='upper left', prop=font)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylim([0, 1])
    # ax1.legend(legends, ['csp_0', 'csp_8', 'csp_3', 'eegnet_0', 'eegnet_8', 'eegnet_3', 'resnet_0', 'resnet_8', 'resnet_3'], loc='upper right')
    ax1.yaxis.grid(True)

    plt.show()

def best_threshold(FPR, TPR, thresholds, stage):
    TNR = 1 - FPR
    balanced_acc = (TPR + TNR) / 2
    # if stage == 'val':
    #     draw_acc(balanced_acc, thresholds)
    max_ind = np.argmax(balanced_acc)
    return thresholds[max_ind], balanced_acc[max_ind]


def _compute_matrics_2(preds, true, stage):

    true = true.cpu().detach()
    preds = preds.cpu().detach()
    FPR, TPR, thresholds = roc_curve(true, preds, pos_label=1)
    auc_score = auc(FPR, TPR)
    # draw_roc(FPR, TPR, auc_score, stage)
    best_thre, balanced_acc = best_threshold(FPR, TPR, thresholds, stage)
    return balanced_acc, auc_score, best_thre


def _compute_matrics_test(preds, true, best_threshold, stage):
    true = true.cpu().detach()
    preds = preds.cpu().detach()
    FPR, TPR, thresholds = roc_curve(true, preds, pos_label=1)
    auc_score = auc(FPR, TPR)
    # draw_roc(FPR, TPR, auc_score, stage)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(preds)):
        if true[i] == 1:
            if preds[i] >= best_threshold:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if preds[i] < best_threshold:
                TN = TN + 1
            else:
                FP = FP + 1
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    balanced_acc = (tpr + tnr) / 2
    return balanced_acc, auc_score


def _compute_acc_test(preds, true):
    acc = []
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(preds)):
            if true[i] == 1:
                if preds[i] >= threshold:
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                if preds[i] < threshold:
                    TN = TN + 1
                else:
                    FP = FP + 1
        tpr = TP / (TP + FN)
        tnr = TN / (TN + FP)
        balanced_acc = (tpr + tnr) / 2
        acc.append(balanced_acc)
    return acc[0], acc[1], acc[2], acc[3]


def _compute_matrics(preds, true, print_tpr=False):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(preds)):
        if true[i] == 0:
            if preds[i] == 0:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if preds[i] == 1:
                TN = TN + 1
            else:
                FP = FP + 1
    tpr = TP / (TP + FN)
    tnr = TN / (TN + FP)
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    except:
        precision = 0
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    balanced_acc = (tpr + tnr) / 2
    if print_tpr:
        print(
            f'TPR: {tpr:.4f}. TNR: {tnr:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. False Positive Rate: {FP / (FP + TN):.4f}.')
    return f1, balanced_acc, precision, recall, FP / (FP + TN)






def _fit(model, train_loader, val_loader, test_loader, testext_loader, class_weight):
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.05)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0)
    lr_max = 0.0005
    lr_min = 0.00001
    max_epoch = 100

    # criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.0)
    # criterion = nn.BCELoss()
    # criterion.to(device)
    model.to(device)
    log_train_loss = []
    log_val_loss = []
    log_test_loss = []
    log_testext_loss = []
    log_train_auc = []
    log_val_auc = []
    log_test_auc = []
    log_testext_auc = []
    log_train_acc = []
    log_val_acc = []
    log_test_acc = []
    log_testext_acc = []
    log_best_threshold = []
    log_test_acc_0 = []  # threshold=0.4, 0.5, 0.6, 0.7
    log_testext_acc_0 = []
    best_val_loss = 100
    for epoch in range(100):
        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=max_epoch, lr_min=lr_min,
                             lr_max=lr_max, warmup=True)
        running_train_loss = 0
        running_val_loss = 0
        running_test_loss = 0
        running_testext_loss = 0
        train_outputs = None
        val_outputs = None
        test_outputs = None
        testext_outputs = None
        preds = []
        true = []


        model.train()
        for i_batch, data_batch in enumerate(train_loader):
            x = data_batch['x'].to(device)
            y = data_batch['y'].to(device)
            sample_weight = data_batch['weight'].to(device)
            sample_weight = torch.reshape(sample_weight, (len(y), 1))
            optimizer.zero_grad()
            outputs = model(x)
            train_loss = func.binary_cross_entropy(outputs, y, weight=sample_weight)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            if train_outputs is None:
                train_outputs = outputs
                train_labels = y
            else:
                train_outputs = torch.cat((train_outputs, outputs), dim=0)
                train_labels = torch.cat((train_labels, y), dim=0)
        preds.append(train_outputs)
        true.append(train_labels)
        # _, train_predicts = torch.max(train_outputs, dim=1)
        # _, train_labels = torch.max(train_labels, dim=1)
        ba_acc, auc_score, _ = _compute_matrics_2(train_outputs, train_labels, stage='train')
        log_train_auc.append(auc_score)
        log_train_acc.append(ba_acc)
        log_train_loss.append(running_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            if val_loader is not None:
                for data_batch in val_loader:
                    x = data_batch['x'].to(device)
                    y = data_batch['y'].to(device)
                    sample_weight = data_batch['weight'].to(device)
                    sample_weight = torch.reshape(sample_weight, (len(y), 1))
                    outputs = model(x)
                    val_loss = func.binary_cross_entropy(outputs, y, weight=sample_weight)
                    running_val_loss += val_loss.item()
                    if val_outputs is None:
                        val_outputs = outputs
                        val_labels = y
                    else:
                        val_outputs = torch.cat((val_outputs, outputs), dim=0)
                        val_labels = torch.cat((val_labels, y), dim=0)
                # _, val_predicts = torch.max(val_outputs, dim=1)
                # _, val_labels = torch.max(val_labels, dim=1)
                preds.append(val_outputs)
                true.append(val_labels)
                ba_acc, auc_score, best_thre = _compute_matrics_2(val_outputs, val_labels, stage='val')
                log_val_auc.append(auc_score)
                log_val_acc.append(ba_acc)
                log_best_threshold.append(best_thre)
                log_val_loss.append(running_val_loss / len(val_loader))

            if test_loader is not None:
                for data_batch in test_loader:
                    x = data_batch['x'].to(device)
                    y = data_batch['y'].to(device)
                    sample_weight = data_batch['weight'].to(device)
                    sample_weight = torch.reshape(sample_weight, (len(y), 1))
                    outputs = model(x)
                    test_loss = func.binary_cross_entropy(outputs, y, weight=sample_weight)
                    running_test_loss += test_loss.item()
                    if test_outputs is None:
                        test_outputs = outputs
                        test_labels = y
                    else:
                        test_outputs = torch.cat((test_outputs, outputs), dim=0)
                        test_labels = torch.cat((test_labels, y), dim=0)
                preds.append(test_outputs)
                true.append(test_labels)
                # _, test_predicts = torch.max(test_outputs, dim=1)
                # _, test_labels = torch.max(test_labels, dim=1)
                threshold_test = log_best_threshold[-1]
                ba_acc_test, auc_test = _compute_matrics_test(test_outputs, test_labels, threshold_test, stage='test')
                ba_acc_04, ba_acc_05, ba_acc_06, ba_acc_07 = _compute_acc_test(test_outputs, test_labels)  # mid: threshold=0.5
                print(f' ACC: [{ba_acc_test:.4f}], 0.4-0.7:[{ba_acc_04:.4f}, '
                      f'{ba_acc_05:.4f}, {ba_acc_06:.4f}, {ba_acc_07:.4f}]')
                log_test_acc_0.append([ba_acc_04, ba_acc_05, ba_acc_06, ba_acc_07])
                log_test_auc.append(auc_test)
                log_test_acc.append(ba_acc_test)
                log_test_loss.append(running_test_loss / len(test_loader))

            if testext_loader is not None:
                for data_batch in testext_loader:
                    x = data_batch['x'].to(device)
                    y = data_batch['y'].to(device)
                    sample_weight = data_batch['weight'].to(device)
                    sample_weight = torch.reshape(sample_weight, (len(y), 1))
                    outputs = model(x)
                    testext_loss = func.binary_cross_entropy(outputs, y, weight=sample_weight)
                    running_testext_loss += testext_loss.item()
                    if testext_outputs is None:
                        testext_outputs = outputs
                        testext_labels = y
                    else:
                        testext_outputs = torch.cat((testext_outputs, outputs), dim=0)
                        testext_labels = torch.cat((testext_labels, y), dim=0)
                preds.append(testext_outputs)
                true.append(testext_labels)
                # _, testext_predicts = torch.max(testext_outputs, dim=1)
                # _, testext_labels = torch.max(testext_labels, dim=1)
                threshold_test = log_best_threshold[-1]
                ba_acc_testext, auc_testext = _compute_matrics_test(testext_outputs, testext_labels, threshold_test, stage='testext')
                ba_acc_testext_04, ba_acc_testext_05, ba_acc_testext_06, ba_acc_testext_07 = _compute_acc_test(
                    testext_outputs, testext_labels)  # mid: threshold=0.5
                log_testext_acc_0.append([ba_acc_testext_04, ba_acc_testext_05, ba_acc_testext_06, ba_acc_testext_07])
                log_testext_auc.append(auc_testext)
                log_testext_acc.append(ba_acc_testext)
                log_testext_loss.append(running_testext_loss / len(testext_loader))




        print(f'epoch: {epoch} '
              f'loss: [{running_train_loss / len(train_loader):.4f} {running_val_loss / len(val_loader):.4f} '
              f'{running_test_loss / len(test_loader):.4f} {running_testext_loss / len(testext_loader):.4f}] '
              f' ba_acc: [{log_train_acc[-1]:.4f} {log_val_acc[-1]:.4f} {log_test_acc[-1]:.4f} {log_testext_acc[-1]:.4f}]'
              f' auc: [{log_train_auc[-1]:.4f} {log_val_auc[-1]:.4f} {log_test_auc[-1]:.4f} {log_testext_auc[-1]:.4f}]'
              f' Threshold: {log_best_threshold[-1]:.4f}')

        # draw_pred(preds, true)

        # if log_val_acc[-1] > best_val_acc:
        if log_val_loss[-1] <= best_val_loss:
            # store results
            # best_val_acc = log_val_acc[-1]
            best_val_loss = log_val_loss[-1]
            out = {
                'acc': log_test_acc[-1],
                'acc_0.4': log_test_acc_0[-1][0],
                'acc_0.5': log_test_acc_0[-1][1],
                'acc_0.6': log_test_acc_0[-1][2],
                'acc_0.7': log_test_acc_0[-1][3],
                'auc': log_test_auc[-1],
                'loss': log_test_loss[-1],
                'acc_ext': log_testext_acc[-1],
                'acc_ext_0.4': log_testext_acc_0[-1][0],
                'acc_ext_0.5': log_testext_acc_0[-1][1],
                'acc_ext_0.6': log_testext_acc_0[-1][2],
                'acc_ext_0.7': log_testext_acc_0[-1][3],
                'auc_ext': log_testext_auc[-1],
                'loss_ext': log_testext_loss[-1],
                'best_thre': log_best_threshold[-1]
            }
            epoch_since_last_best = 0
        else:
            epoch_since_last_best = epoch_since_last_best + 1

        # Early stopping
        if len(log_val_loss) > 29:
            vals = np.array(log_val_loss[-10:])
            trains = np.array(log_train_loss[-10:])
            if ((np.amax(vals) - np.amin(vals)) < 0.0005) or ((np.amax(trains) - np.amin(trains)) < 0.0002) or (
                    epoch_since_last_best > 20):
                print('Triggered early stopping.')
                break

    print(out)
    # print(f'Finished! TEST ACC: {ba_acc_test:.4f} PRECISION: {precision_test:.4f} RECALL: {recall_test:.4f} F1: {f1_test:.4f}')
    # out = {
    #     'acc': ba_acc_test,
    #     'prec': precision_test,
    #     'recall': recall_test,
    #     'f1': f1_test,
    #     'loss': log_test_loss[-1],
    #     'acc_ext': ba_acc_testext,
    #     'prec_ext': precision_testext,
    #     'recall_ext': recall_testext,
    #     'f1_ext': f1_testext,
    #     'loss_ext': log_testext_loss[-1],
    #     'fp_over_allp': fp_over_allp
    # }
    # ax_loss = fig.add_subplot(121, title="Loss")
    # ax_acc = fig.add_subplot(122, title="ACC")
    # ax_loss.set_xlim([0, 200])
    # ax_acc.set_xlim([0, 200])
    # ax_loss.set_ylim([0, 1.5])
    # ax_acc.set_ylim([0, 1])
    # ax_loss.grid()
    # ax_acc.grid()
    #
    # ax_loss.plot(log_train_loss, label='Train')
    # ax_loss.plot(log_val_loss, label='Val')
    # ax_loss.plot(log_test_loss, label='Test')
    #
    # ax_acc.plot(log_train_acc, label='Train')
    # ax_acc.plot(log_val_acc, label='Val')
    # ax_acc.plot(log_test_acc, label='Test')
    #
    # ax_loss.legend()
    # ax_acc.legend()
    #
    # plt.show()
    return model, out


def _model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        # total_params += param
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print(f"Total Params:{total_params}")
