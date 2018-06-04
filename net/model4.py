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
		self.prelu = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
		self.convdepthwise2=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=(1,1),bias=False)
		self.convpointwise2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False)
		self.convdepthwise3=nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(5,1),stride=1,padding=(2,0),bias=False)
		self.convpointwise3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False)
		self.convdepthwise5=nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),stride=1,padding=1,bias=False)
		self.convpointwise5=nn.Conv2d(in_channels=64,out_channels=upscale_factor**2,kernel_size=1,stride=1,padding=0,bias=False)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
		self._initialize_weights()
	def forward(self, x):
		x = self.prelu(self.conv1(x))
		x = self.prelu(self.convpointwise2(self.convdepthwise2(x)))
		r1=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r2=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r3=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r4=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r5=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r6=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r7=x
		x = self.prelu(self.convpointwise3(self.convdepthwise3(x)))
		r8=x
		x+=r1
		x+=r2
		x+=r3
		x+=r4
		x+=r5
		x+=r6
		x+=r7
		x+=r8
		x = self.pixel_shuffle(self.convpointwise5(self.convdepthwise5(x)))
		return x
	def _initialize_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convdepthwise2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise2.weight)
		init.orthogonal_(self.convdepthwise3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise3.weight)
		init.orthogonal_(self.convdepthwise5.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise5.weight)
