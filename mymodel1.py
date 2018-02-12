import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu6 = nn.ReLU6()
        self.relu=nn.ReLU()
        self.conv1 = nn.Conv2d(1,32, (3, 3), (1, 1), (1, 1))
        self.conv2= nn.Conv2d(32,64, (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.conv4= nn.Conv2d(128,64, (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x=self.relu6(self.conv2(x))
        x = self.conv3(x)
        x=self.conv4(x)
        x = self.relu6(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))

        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight)
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight)
        init.orthogonal(self.conv5.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv6.weight)