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
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
		self.convdepthwise2=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=(1,1),bias=False)
		self.convpointwise2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False)
		self.convdepthwise3=nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(5,1),stride=1,padding=(2,0),bias=False)
		self.convpointwise3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,bias=False)
		self.convdepthwise4=nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(5,1),stride=1,padding=(2,0),bias=False)
		self.convpointwise4=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1,padding=0,bias=False)
		self.convdepthwise5=nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=1,padding=1,bias=False)
		self.convpointwise5=nn.Conv2d(in_channels=32,out_channels=upscale_factor**2,kernel_size=1,stride=1,padding=0,bias=False)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
		self._initialize_weights()


	def forward(self, x):
		x = self.relu(self.conv1(x))
		residual=x
		x = self.relu(self.convpointwise2(self.convdepthwise2(x)))
		x = self.relu(self.convpointwise3(self.convdepthwise3(x)))
		x = self.relu(self.convpointwise3(self.convdepthwise3(x)))
		x = self.relu(self.convpointwise4(self.convdepthwise4(x)))
		x+=residual
		x = self.pixel_shuffle(self.convpointwise5(self.convdepthwise5(x)))
		return x
	def _initialize_weights(self):
		init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convdepthwise2.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise2.weight)
		init.orthogonal_(self.convdepthwise3.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise3.weight)
		init.orthogonal_(self.convdepthwise4.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise4.weight)
		init.orthogonal_(self.convdepthwise5.weight, init.calculate_gain('relu'))
		init.orthogonal_(self.convpointwise5.weight)

		'''
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n)).clamp_(min=0,max=2)
		'''
