import torch
import torch.nn as nn
import torch.nn.init as init

# this 5x1 conv 3x3 conv
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1),bias=False)
        self.convdw2=nn.Conv2d(16,16,(3,3),1,(1,1),groups=16,bias=False)
        self.convpw2=nn.Conv2d(16,32,(1,1),1,(0,0),bias=False)

        self.convdw3=nn.Conv2d(32,32,(3,3),1,(1,1),groups=32,bias=False)
        self.convpw3=nn.Conv2d(32,32,(1,1),1,0,bias=False)

        self.convdw4=nn.Conv2d(32,32,(3,3),1,(1,1),groups=32,bias=False)
        self.convpw4=nn.Conv2d(32,16,(1,1),1,0,bias=False)

        self.convdw5=nn.Conv2d(16,16,(3,3),1,(1,1),groups=16,bias=False)
        self.convpw5=nn.Conv2d(16,upscale_factor**2,(1,1),1,0,bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.convdw2(x)
        x = self.convpw2(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x = self.convdw5(self.relu(x))
        x = self.convpw5(x)
        x = self.pixel_shuffle(x)
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.convdw2.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw2.weight)

        init.orthogonal(self.convdw3.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw3.weight)

        init.orthogonal(self.convdw4.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw4.weight)
        init.orthogonal(self.convdw5.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw5.weight)
