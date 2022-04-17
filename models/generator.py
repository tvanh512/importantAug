import torch
import torch.nn.functional as F
import torch.nn as nn
from tools.data_utils import get_same_padding


class MaskGen_CNN2(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1):
        super(MaskGen_CNN2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,in_channels *2,kernel_size,padding = get_same_padding(kernel_size, stride, dilation))
        self.conv2 = torch.nn.Conv2d(in_channels *2,in_channels *2,kernel_size,padding = get_same_padding(kernel_size, stride, dilation))
        self.conv3 = torch.nn.Conv2d(in_channels *2,in_channels *2,kernel_size,padding = get_same_padding(kernel_size, stride, dilation))
        self.conv4 = torch.nn.Conv2d(in_channels *2,out_channels,kernel_size,padding = get_same_padding(kernel_size, stride, dilation))
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.sigmoid(x)
        return x

class MaskGen_CNN(nn.Module):
    def __init__(self,
                num_layers,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1):
        super(MaskGen_CNN, self).__init__()
        self.convs = nn.Sequential(*[torch.nn.Conv2d(in_channels,out_channels,kernel_size,padding = get_same_padding(kernel_size, stride, dilation)) for _ in range(num_layers)])
    def forward(self, x):
        x = self.convs(x)
        x = F.sigmoid(x)
        return x
        