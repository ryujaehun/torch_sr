from __future__ import print_function
import argparse,random
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.data import get_training_set, get_test_set
from torch.nn.modules.module import _addindent
from pandas import DataFrame
import pandas as pd
from utils.utils import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor', type=int,  default=2, help="super resolution upscale factor")
parser.add_argument('--model', type=str, default='weight1', help='choosing a model')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.003, help='Learning Rate. Default=0.01')
parser.add_argument('--batchSize', type=int, default=12, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--cuda', action='store_true' , default=True,help='use cuda?')
parser.add_argument('--depthwise', type=bool , default=False,help='use depthwise model?')




opt = parser.parse_args()
if opt.depthwise is True:
    from net.model import Net
else:
    from net.model_dw import Net

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(random.randint(1,1000))
if cuda:
    torch.cuda.manual_seed(random.randint(1,1000))


print('===> Loading datasets')

train_set = get_training_set(opt.upscale_factor,opt.data)
test_set = get_test_set(opt.upscale_factor,opt.data)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Loadding model')
train=Net(opt.upscale_factor)
model=torch.load(opt.model)
keys=model.keys()
criterion = nn.MSELoss()

pruning=100
for _ in range(1,pruning):
    lista=cal_pruning(train,model,epoch=1)
    indexa=rank(lista)
    _pruning(train,model,indexa,epoch=1)
    print(_," 번째 pruning 완료")
