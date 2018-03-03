import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
path=os.getcwd()
parser = argparse.ArgumentParser(description='pth to dict')
parser.add_argument('--path', type=str, required=True, help="model path")
parser.add_argument('--name', type=str,default='model' ,required=False, help="model name")
opt = parser.parse_args()
model=torch.load(opt.path)
temp=model.state_dict()
torch.save(temp,os.path.join(path,opt.name))
print('saving Complete',os.path.join(path,opt.name))
