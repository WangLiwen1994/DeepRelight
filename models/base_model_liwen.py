import torch
import torch.nn as nn
from torch.nn import functional as F
##from deform_conv_v2 import DeformConv2d

class UpBlock(torch.nn.Module):
    def __init__(self, input_size):
        super(UpBlock, self).__init__()
        output_size = input_size//2
        self.conv1 = DeconvBlock(input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size=4, stride=2, padding=1, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size):
        super(DownBlock, self).__init__()
        output_size = input_size*2
        self.conv1 = ConvBlock(input_size, output_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size=3, stride=2, padding=1, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.mergeFeather = ConvBlock(inchannel, inchannel, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.mergeFeather(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=True, groups=1):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias,groups=groups)
        if self.isuseBN:
            self.bn = nn.InstanceNorm2d(output_size)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, isuseBN=True, bias=True):
        super(DeconvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.InstanceNorm2d(output_size)

        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.deconv(x)
        if self.isuseBN:
            out = self.bn(out)
        return self.act(out)

