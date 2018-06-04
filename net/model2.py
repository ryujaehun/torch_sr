import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1))
        self.convdw2=nn.Conv2d(32,32,(3,3),1,1,groups=32)
        self.convpw2=nn.Conv2d(32,64,1,1,0)
        self.convdw3=nn.Conv2d(64,64,(3,3),1,1,groups=64)
        self.convpw3=nn.Conv2d(64,64,1,1,0)
        self.convdw4=nn.Conv2d(64,64,(3,3),1,1,groups=64)
        self.convpw4=nn.Conv2d(64,upscale_factor**2,1,1,0)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.convdw2(x)
        x = self.convpw2(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x = self.pixel_shuffle(x)
        return x
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convdw2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convpw2.weight)
        init.orthogonal_(self.convdw3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convpw3.weight)
        init.orthogonal_(self.convdw4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convpw4.weight)
