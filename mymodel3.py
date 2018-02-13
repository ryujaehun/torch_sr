import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2))
        self.convdw2=nn.Conv2d(32,32,3,1,1,groups=32,bias=False)
        self.convpw2=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw3=nn.Conv2d(32,32,3,1,1,groups=32,bias=False)
        self.convpw3=nn.Conv2d(32,64,1,1,0,bias=False)

        self.convdw4=nn.Conv2d(64,64,3,1,1,groups=64,bias=False)
        self.convpw4=nn.Conv2d(64,64,1,1,0,bias=False)

        self.convdw5=nn.Conv2d(64,64,3,1,1,groups=64,bias=False)
        self.convpw5=nn.Conv2d(64,128,1,1,0,bias=False)

        self.convdw6=nn.Conv2d(128,128,3,1,1,groups=128,bias=False)
        self.convpw6=nn.Conv2d(128,64,1,1,0,bias=False)

        self.convdw7=nn.Conv2d(64,64,3,1,1,groups=64,bias=False)
        self.convpw7=nn.Conv2d(64,32,1,1,0,bias=False)


        self.conv8 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.convdw2(x))
        x = self.convpw2(x)
        x = self.relu(self.convdw3(x))
        x = self.convpw3(x)
        x = self.relu(self.convdw4(x))
        x = self.convpw4(x)
        x = self.relu(self.convdw5(x))
        x = self.convpw5(x)
        x = self.relu(self.convdw6(x))
        x = self.convpw6(x)
        x = self.relu(self.convdw7(x))
        x = self.convpw7(x)
        x = self.pixel_shuffle(self.conv8(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convdw2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw2.weight)
        init.orthogonal(self.convdw3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw3.weight)
        init.orthogonal(self.convdw4.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw4.weight)
        init.orthogonal(self.convdw5.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw5.weight)
        init.orthogonal(self.convdw6.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw6.weight)
        init.orthogonal(self.convdw7.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convpw7.weight)
        init.orthogonal(self.conv8.weight)
