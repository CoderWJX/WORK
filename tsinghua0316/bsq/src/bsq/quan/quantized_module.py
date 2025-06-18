import torch
from torch import nn


class RescaleReLULayer(torch.nn.Module):
    def __init__(self, scale, zeropoint=None, out_type='uint8'):
        super(RescaleReLULayer, self).__init__()
        assert zeropoint == None or scale.shape == zeropoint.shape
        self.scale = nn.Parameter(scale)
        if zeropoint == None:
            self.zeropoint = nn.Parameter(torch.zeros_like(scale))
        else:
            self.zeropoint = nn.Parameter(zeropoint)
        if out_type == 'uint8':
            self.thd_neg = 0
            self.thd_pos = 255
        elif out_type == 'int8':
            self.thd_neg = -128
            self.thd_pos = 127
        elif out_type == 'int16':
            self.thd_neg = -32768
            self.thd_pos = 32767
        elif out_type == 'uint16':
            self.thd_neg = 0
            self.thd_pos = 65535
        else:
            raise NotImplementedError

    def forward(self, x):
        return torch.clamp(torch.round(self.scale * x + self.zeropoint), self.thd_neg, self.thd_pos)


class QuanConv2dRescaleReLU(torch.nn.Module):
    def __init__(self, m: torch.nn.Conv2d, scale, zeropoint, out_type='uint8'):
        super(QuanConv2dRescaleReLU, self).__init__()
        self.conv = torch.nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size,
                                    stride=m.stride,
                                    padding=m.padding,
                                    dilation=m.dilation,
                                    groups=m.groups,
                                    bias=None,
                                    padding_mode=m.padding_mode)
        self.rescale = RescaleReLULayer(scale, zeropoint, out_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.rescale(x)

        return x


class QuanLinearRescaleReLU(torch.nn.Module):
    def __init__(self, m: torch.nn.Linear, scale, zeropoint, out_type='int16', input_type='uint8'):
        super(QuanLinearRescaleReLU, self).__init__()
        self.fc = torch.nn.Linear(m.in_features, m.out_features, bias=False)
        self.input_type = input_type
        self.rescale = RescaleReLULayer(scale, zeropoint, out_type)

    def forward(self, x):
        if self.input_type == 'uint8':
            x = torch.clamp(x, 0, 255)
        x = self.fc(x)
        x = self.rescale(x)
        return x
