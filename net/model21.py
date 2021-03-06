import torch
import torch.nn as nn
import torch.nn.init as init

# addition residual connection
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1),bias=False)
        self.convdw2=nn.Conv2d(16,16,(5,1),1,(2,0),groups=16,bias=False)
        self.convpw2=nn.Conv2d(16,32,(1,1),1,(0,0),bias=False)

        self.convdw3=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw3=nn.Conv2d(32,32,(1,1),1,0,bias=False)

        self.convdw4=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw4=nn.Conv2d(32,16,(1,1),1,0,bias=False)

        self.conv5 = nn.Conv2d(16, upscale_factor ** 2, (3, 3), (1, 1), (1, 1),bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual=x
        x = self.convdw2(self.relu(x))
        x = self.convpw2(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x+=residual
        x = self.pixel_shuffle(self.conv5(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convdw2.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw2.weight)

        init.orthogonal(self.convdw3.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw3.weight)

        init.orthogonal(self.convdw4.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw4.weight)
        init.orthogonal(self.conv5.weight)
