#! /usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import torch.nn.init as init
class Net(nn.Module):
    def __init__(self,upscale_factor):
        super(Net, self).__init__()
        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.convdepthwise2=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(5,1),stride=1,padding=(2,0),bias=False)
        self.convpointwise2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1,padding=0,bias=False)
        self.convdepthwise3=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.convpointwise3=nn.Conv2d(in_channels=32,out_channels=upscale_factor**2,kernel_size=1,stride=1,padding=0,bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()
    def forward(self, x):
        x = self.prelu(self.conv1(x))
        residual1=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual2=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual3=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual4=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual5=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual6=x
        x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
        residual7=x
        x+=residual1
        x+=residual2
        x+=residual3
        x+=residual4
        x+=residual5
        x+=residual6
        x+=residual7
        x = self.pixel_shuffle(self.convpointwise3(self.convdepthwise3(x)))
        return x
    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convdepthwise2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpointwise2.weight)
        init.orthogonal(self.convdepthwise3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpointwise3.weight)
