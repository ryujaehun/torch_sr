import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (9, 9), (1, 1))
        self.conv2 = nn.Conv2d(64, 32, (1, 1), (1, 1))    
        self.convdw3=nn.Conv2d(32,32,5,1,groups=32,bias=True)
        self.convpw3=nn.Conv2d(32,1,1,1,0,bias=True)


        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.convdw3(x)
        x = self.convpw3(x)
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight)
        init.orthogonal(self.convdw3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw3.weight, init.calculate_gain('relu'))
