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
from utils.metric import psnr,ssim,ms_ssim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model_name', type=str,default='model_epoch_400.pth' ,required=False, help='model file to use')
parser.add_argument('--output_filename', default='result',type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', required=False, help='use cuda')
parser.add_argument('--dataset', default='Set5', type=str,required=False, help='use cuda')
parser.add_argument('--model', default='01', type=str,required=False, help='model')

opt = parser.parse_args()
model=torch.load(opt.model_name)
def psnr(original_image, new_image):
    original_arr = np.array(original_image)
    new_arr = np.array(new_image)
    height, width, foo = original_arr.shape
    height2, width2, foo2 = new_arr.shape
    if height != height2 or width != width2:
        raise Exception('Images must have the same size')
    MSE = 0
    for i in range(height):
        for j in range(width):
            # only green channel
            MSE += (int(original_arr[i][j][1]) - int(new_arr[i][j][1]))**2
    MSE = float(MSE) / float(width*height)
    if MSE == 0:
        pass
        #print 'Same image!'
    else:
        MAX = 255
        PSNR = 10 * math.log10(float(MAX**2)/float(MSE))
        return PSNR  # el resultado en decibeles

print(opt)
curr=os.getcwd()
os.chdir(os.path.join(curr,'dataset/data/'+opt.dataset+'/image_SRF_2'))
for i in os.listdir():
    if i[5:7]== opt.model:
        if i[14:16]== "LR":
            Flag=False
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

print(' BICUBIC PSNR is ',psnr(img1,img_hr))
print(' Our PSNR is ',psnr(out_img,img_hr))

os.chdir(curr)
