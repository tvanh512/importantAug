import torch.nn as nn
import torch.nn.functional as F
from models.cnn import Conv
import torchaudio
import torch

class Classifier_CNN(nn.Module):
    def __init__(self,n_class,n_feat,n_layer,kernel_size,dropout=0.0):
        print('init Classifier CNN')
        super(Classifier_CNN, self).__init__()
        self.convs = nn.Sequential(*[Conv(n_feat,n_feat,kernel_size) for _ in range(n_layer)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1=  nn.Linear(n_feat,n_feat)
        self.linear2 = nn.Linear(n_feat,n_feat)
        self.linear3 = nn.Linear(n_feat,n_class)
        self.dropout = nn.Dropout(dropout)
        self.amptodB_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude')
    
    def forward(self,x):
        if not torch.is_complex(x):
            x = torch.view_as_complex(x)
        x = x.squeeze()
        xdB = self.amptodB_transform(x.abs()) # xdB has shape B x n_feat x Time, where n_feat has similar meaning as frequency
        x = self.convs(xdB)
        B,C,T = x.size()
        x = self.pool(x)
        x = x.view(B,C)
        x = F.selu(self.linear1(x))
        x = F.selu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        x = x.unsqueeze(1)    
        return F.log_softmax(x, dim=2)
