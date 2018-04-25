import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (1, 1), (1, 1), (0, 0))

        self.convdw2=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw2=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw3=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw3=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw4=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw4=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw5=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw5=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw6=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw6=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw7=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw7=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw8=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw8=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw9=nn.Conv2d(32,32,(5,1),1,(2,0),groups=32,bias=False)
        self.convpw9=nn.Conv2d(32,32,1,1,0,bias=False)

        self.convdw10=nn.Conv2d(32,32,(3,3),1,(1,1),groups=32,bias=False)
        self.convpw10=nn.Conv2d(32,upscale_factor**2,1,1,0,bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual=x
        re=x
        x = self.convdw2(self.relu(x))
        x = self.convpw2(x)
        x = self.convdw2(self.relu(x))
        x = self.convpw2(x)
        x+=re
        re=x
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)
        x = self.convdw3(self.relu(x))
        x = self.convpw3(x)

        x+=re
        re=x
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x+=re

        re=x
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x = self.convdw4(self.relu(x))
        x = self.convpw4(x)
        x+=re

        re=x
        x = self.convdw5(self.relu(x))
        x = self.convpw5(x)
        x = self.convdw5(self.relu(x))
        x = self.convpw5(x)
        x+=re

        re=x
        x = self.convdw6(self.relu(x))
        x = self.convpw6(x)
        x = self.convdw6(self.relu(x))
        x = self.convpw6(x)
        x+=re

        re=x
        x = self.convdw7(self.relu(x))
        x = self.convpw7(x)
        x = self.convdw7(self.relu(x))
        x = self.convpw7(x)
        x+=re

        re=x
        x = self.convdw8(self.relu(x))
        x = self.convpw8(x)
        x = self.convdw8(self.relu(x))
        x = self.convpw8(x)
        x+=re

        re=x
        x = self.convdw9(self.relu(x))
        x = self.convpw9(x)
        x = self.convdw9(self.relu(x))
        x = self.convpw9(x)
        x+=re

        x+=residual
        x = self.convdw10(self.relu(x))
        x = self.convpw10(x)
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
        init.orthogonal(self.convdw6.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw6.weight)
        init.orthogonal(self.convdw7.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw7.weight)
        init.orthogonal(self.convdw8.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw8.weight)
        init.orthogonal(self.convdw9.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw9.weight)
        init.orthogonal(self.convdw10.weight,init.calculate_gain('relu'))
        init.orthogonal(self.convpw10.weight)
