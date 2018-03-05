from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from net.model10 import Net
import numpy as np
import os
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model_name', type=str,default='model_epoch_399.pth' ,required=False, help='model file to use')
parser.add_argument('--output_filename', default='result',type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', required=False, help='use cuda')
parser.add_argument('--dataset', default='Set5', type=str,required=False, help='use cuda')
parser.add_argument('--model', default='1', type=str,required=False, help='model')

opt = parser.parse_args()
model=torch.load(opt.model_name)
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

print(opt)
curr=os.getcwd()
os.chdir(os.path.join(curr,'dataset/data/'+opt.dataset+'/image_SRF_2'))
for i in os.listdir():
    if i[6]== opt.model:
        if i[14:16]== "LR":
            img = Image.open(i).convert('YCbCr')
            img1 = Image.open(i)
        else:
            img_hr=Image.open(i)
y, cb, cr = img.split()
input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
if opt.cuda:
    model = model.cuda()
    input = input.cuda()
out = model(input)
out = out.cpu()
out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

img1=img1.resize(out_img_y.size, Image.BICUBIC)
out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
print(' Our PSNR is ',psnr(np.array(out_img),np.array(img_hr)))

print(' BICUBIC PSNR is ',psnr(np.array(img1),np.array(img_hr)))
os.chdir(curr)
