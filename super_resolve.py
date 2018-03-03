from __future__ import print_function
import argparse

import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from net.model10 import Net
import numpy as np
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str,default='test.bmp' ,required=False, help='input image to use')
parser.add_argument('--model_name', type=str,default='model' ,required=False, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', required=True, help='use cuda')
opt = parser.parse_args()
print(opt)
img = Image.open(opt.input_image).convert('YCbCr')
y, cb, cr = img.split()
model=Net(2)
model.load_state_dict(torch.load(opt.model_name))
#model=torch.load("model_epoch_400.pth")

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
out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
