from __future__ import print_function
import argparse
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import datetime,random,os,csv
from torch.nn.modules.module import _addindent
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor', type=int,  default=2, help="super resolution upscale factor")
parser.add_argument('--data', type=str, default='BSDS300',required=False, help="train data path")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=40, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true'  default=True,help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--model', type=int, default='2', help='choose a model')
