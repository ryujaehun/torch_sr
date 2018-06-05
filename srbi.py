
from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import os,glob
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input','-i',default='Animals' ,type=str, required=False, help='input image to use')
parser.add_argument('-u',type=int,default=2, required=False, help='use cuda')
opt = parser.parse_args()
join=os.path.join
root=os.getcwd()
result=join(root,'vresult')
result=join(result,opt.input)
dataset=join(root,'dataset')
datasetfolder=join(dataset,opt.input)
videolist=glob.glob(datasetfolder+'/*.bmp')

videolist.sort()
print(opt)
if not os.path.exists(result):
    os.makedirs(result)
for i in videolist:
    img = Image.open(join(datasetfolder,i)).convert('YCbCr')
    y, cb, cr = img.split()
    #model = torch.load(opt.model)
    #input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    #if opt.cuda:
    #    model = model.cuda()
    #    input = input.cuda()
    #out = model(input)
    #out = out.cpu()
    #out_img_y = out.data[0].numpy()
    #out_img_y *= 255.0
    #out_img_y = out_img_y.clip(0, 255)
    #out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_y=y.resize((y.size[0]*opt.u,y.size[1]*opt.u), Image.BICUBIC)
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(join(result,i.split('/')[-1]))

