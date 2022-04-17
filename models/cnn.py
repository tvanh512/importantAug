import torch.nn as nn
from tools.data_utils import get_same_padding

class Conv(nn.Module):    
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        dropout=0,
    ):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size,
            groups=in_channels,
            padding = get_same_padding(kernel_size, stride, dilation)
            ) 
        
        self.pointwise = nn.Conv1d(
            in_channels = in_channels,
            out_channels =  out_channels,
            kernel_size=1)
        
        self.activation = nn.Sequential(
            nn.SELU(),
            nn.Dropout(dropout))
        
    def forward(self, x):
        #x before depthwide torch.Size([B, F, T])
        #out.shape after depthwide torch.Size([B,F,T])
        #out.shape after pointwise torch.Size([B,F,T])
        out = self.depthwise(x) 
        out= self.pointwise(out)
        out = self.activation(out) + x
        return out