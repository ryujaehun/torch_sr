#!/usr/bin/python3
from skimage import img_as_float
from skimage.measure import compare_psnr #as psnr
from skimage.measure import compare_ssim #as ssim
def psnr(img1,img2,uf=2):
    width, height = img1.size
    return compare_psnr(img_as_float(img1.crop((0,0,width-uf,height-uf))),img_as_float(img2.crop((0,0,width-uf,height-uf))))
def ssim(img1,img2,uf=2):
    width, height = img1.size
    return compare_ssim(img_as_float(img1.crop((0,0,width-uf,height-uf))),img_as_float(img2.crop((0,0,width-uf,height-uf))),multichannel=True)
