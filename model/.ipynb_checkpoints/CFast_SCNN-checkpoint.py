###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################

"""Fast Segmentation Convolutional Neural Network"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CFastSCNN']


class CFastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(CFastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        #print(size)
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        #print(x.shape)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return x

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, in_chanels=64):
        super(AttentionModule, self).__init__()
        self.cam = ChannelAttentionModule(in_channels=64)#通道注意力模块
        self.sam = SpatialAttentionModule(in_channels=64)#空间注意力模块

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, r=0.5):
        super(ChannelAttentionModule, self).__init__()
        #全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)
        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * r)),
            nn.ReLU(),
            nn.Linear(int(in_channels * r), in_channels),
            nn.Sigmoid()
        )

        #全局最大池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * r)),
            nn.ReLU(),
            nn.Linear(int(in_channels * r), in_channels),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #最大池化分支
        max_bench = self.MaxPool(x)
        # print("maxpool:{}".format(max_bench.shape))
        # print("x:{}".format(x.shape))
        #送入全连接神经网络MLP，得到权重
        max_in = max_bench.view(max_bench.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        #2.全局池化分支
        avg_bench = self.MaxPool(x)
        # 送入全连接神经网络MLP，得到权重
        avg_in = avg_bench.view(avg_bench.size(0), -1)
        avg_weight = self.fc_MaxPool(avg_in)

        #maxpool + avgpool的权重得到总的weight，然后经过sigmoid激活
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        #将维度b，c的weight进行reshape，变为（h，w，1，1）的形式与输入x相乘
        h, w =weight.shape
        #print(weight.shape)
        #通道注意力参数Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))
        #print(Mc.shape)
        #print(Mc)
        #乘积获得结果
        x = Mc * x
        #print(x.shape)

        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #将x维度为[N,C,H,W]沿着维度C进行操作，所以dim=1，结果为[N,H,W]
        MaxPool = torch.max(x, dim=1)
        AvgPool = torch.mean(x, dim=1)
        #print(MaxPool.values)
        #print(AvgPool)

        #增加维度，变为[N,1,H,W]
        MaxPool = torch.unsqueeze(MaxPool.values, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        #维度拼接[N,2,H,W],获得特征图
        x_cat = torch.concat((MaxPool, AvgPool), dim=1)

        #进行卷积操作得到空间注意力结果
        x_out = self.conv2d(x_cat)
        #print(x_out.shape)
        Ms = self.sigmoid(x_out)
        #print(Ms.shape)
        #print(Ms)

        #与输入x相乘得到空间注意力结果
        x = Ms * x
        #print(x)

        return x

class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out2 = nn.Sequential(
            # nn.AdaptiveAvgPool2d(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        #第三个分支的注意力
        self.attetion = AttentionModule(64)
        self.relu = nn.ReLU(True)

        self.inter_channel = out_channels // 2
        self.conv_q = nn.Conv2d(in_channels=out_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_k = nn.Conv2d(in_channels=out_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.conv_v = nn.Conv2d(in_channels=out_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.adaptivepool_k = nn.AdaptiveAvgPool2d((8, 8))
        self.adaptivepool_v = nn.AdaptiveAvgPool2d((8, 8))
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)


    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        #print("higher_res_feature:{}".format(higher_res_feature.shape))
        attention_feature = self.attetion(higher_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        out1 = torch.cat([attention_feature, lower_res_feature], dim=1)
        # out1 = self.conv_out1(out1)
        # self.maxpool = nn.AdaptiveMaxPool2d((out1.size(2), out1.size(3)))
        # out2 = self.maxpool(out1)
        # out2 = self.conv_out2(out2)
        # out3 = out1 * out2


        #print("concat结果：{}".format(out1.shape))
        out = higher_res_feature + out1
        # out = higher_res_feature + lower_res_feature
        #ASAM模块
        # [N, C, H , W]
        b, c, h, w = out.size()
        # print(x.size())
        # [N, C/2, H * W]
        x_q = self.conv_q(out).view(b, c, -1).permute(0, 2, 1)
        # print("x_phi.shape:{}".format(x_q.shape))
        # [N, H * W, C/2]
        x_k = self.conv_k(out)
        # print(x_k.shape)
        x_k = self.adaptivepool_k(x_k)
        # print(x_k.shape)
        x_k = x_k.view(b, c, -1).contiguous()
        x_v = self.conv_v(out)
        x_v = self.adaptivepool_v(x_v).view(b, c, -1).permute(0, 2, 1).contiguous()
        # print("x_k_shape:{}".format(x_k.shape))
        # print("x_v_shape:{}".format(x_v.shape))
        # [N, H * W, H * W]
        mul_q_k = torch.matmul(x_q, x_k)
        mul_q_k = self.softmax(mul_q_k)
        # print("mul_q_k_shape:{}".format(mul_q_k.shape))
        # [N, H * W, C/2]
        mul_q_k_v = torch.matmul(mul_q_k, x_v)
        # [N, C/2, H, W]
        mul_q_k_v = mul_q_k_v.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # print("mul_q_k_v:{}".format(mul_q_k_v.shape))
        # [N, C, H , W]
        mask = self.conv_mask(mul_q_k_v)
        out = mask + out
        return self.relu(out)



class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        #print(x.shape)
        return x


# def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
#     acronyms = {
#         'pascal_voc': 'voc',
#         'pascal_aug': 'voc',
#         'ade20k': 'ade',
#         'coco': 'coco',
#         'citys': 'citys',
#     }
#     from data_loader import datasets
#     model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
#     if pretrained:
#         if(map_cpu):
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
#         else:
#             model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
#     return model


if __name__ == '__main__':
    img = torch.randn(2, 3, 1024, 2048)
    # model = get_fast_scnn('citys')
    # outputs = model(img)
    # print(outputs)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # # for parameter in parameters:
    # #     print(np.prod(parameter.size()) / 1_000_000)
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # print('Trainable Parameters: %.3fM' % parameters)
