from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image,ImageFont, ImageDraw
from torchvision.transforms import ToTensor
import numpy as np
import os
import math
from utils.metric import psnr,ssim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model_name', type=str,default='model_epoch_400.pth' ,required=False, help='model file to use')
parser.add_argument('--output_filename', default='result',type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true',  default=True,required=False, help='use cuda')
parser.add_argument('--upscale_factor', default='2', type=str,required=False, help='model')
parser.add_argument('--visual','-v' ,default=1, type=int,required=False, help='print psnr and ssim')
#parser.add_argument('--dataset', default='Set5', type=str,required=False, help='use cuda')
#parser.add_argument('--model', default='01', type=str,required=False, help='model')
def inference(epoch,savepath,datapath,name,dataset):
    '''
    input:
    epoch: print 목적을 위하여
    savepath: 알맞은 폴더에 저장을 위해
    name: data의 순서 000~100
    dataset:data의 이름 ex) Set5
    '''
    global model
    # 총 3개를 연다
    img = Image.open(os.path.join(datapath,name)).convert('YCbCr')
    img_bicubic = Image.open(os.path.join(datapath,name))
    img_hr=Image.open(os.path.join(datapath,name.replace('LR',"HR")))
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
    img_bicubic=img_bicubic.resize(out_img_y.size, Image.BICUBIC)
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    #여기까지 inference code
    # psnr 및 ssim 을 구해서 pillow 에 draw 한다.
    global matrix
    img=img.convert('RGB')
    if dataset is "Set14" and epoch is 3:
        img_bicubic=img_bicubic.convert('RGB')
        img_hr=img_hr.convert('RGB')
    matrix=[sum(x) for x in zip(matrix, [psnr(img_bicubic,img_hr),psnr(out_img,img_hr),ssim(img_bicubic,img_hr),ssim(out_img,img_hr)])]
    # 0: BICUBIC PSNR 1: SR PSNR 2: BICUBIC SSIM 3: SR SSIM
    if opt.visual is 2:
        print(dataset,i,'th')
        print(' BICUBIC PSNR is ',psnr(img_bicubic,img_hr))
        print(' Our PSNR is ',psnr(out_img,img_hr))
        print(' BICUBIC SSIM is ',ssim(img_bicubic,img_hr))
        print(' Our SSIM is ',ssim(out_img,img_hr))
    font = ImageFont.truetype("arial.ttf", 20)
    draw = ImageDraw.Draw(img_bicubic)
    draw.text((0, 0), "BICUBIC",font=font,fill=(0,0,0,255))
    draw.text((0, 20), "SSIM:"+str(ssim(img_bicubic,img_hr)),font=font,fill=(0,0,0,255))
    draw.text((0, 40), "PSNR:"+str(psnr(img_bicubic,img_hr)),font=font,fill=(0,0,0,255))
    img_bicubic.save(os.path.join(savepath,dataset+"_"+name[0:13]+'_bicubic.png'),"PNG")
    draw = ImageDraw.Draw(out_img)
    draw.text((0, 0), "OURS",font=font,fill=(0,0,0,255))
    draw.text((0, 20), "SSIM:"+str(ssim(out_img,img_hr)),font=font,fill=(0,0,0,255))
    draw.text((0, 40), "PSNR:"+str(psnr(out_img,img_hr)),font=font,fill=(0,0,0,255))
    out_img.save(os.path.join(savepath,dataset+"_"+name[0:13]+'_superResolution.png'),"PNG")
    draw = ImageDraw.Draw(img_hr)
    draw.text((0, 0), "Ground True High Resolution",font=font,fill=(0,0,0,255))
    draw.text((0, 20), "Size:"+str(img_hr.size[0])+" x "+str(img_hr.size[1]),font=font,fill=(0,0,0,255))
    img_hr.save(os.path.join(savepath,dataset+"_"+name[0:13]+'_HR.png'),"PNG")
    img=img.convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "Ground True Low Resolution",font=font,fill=(0,0,0,255))
    draw.text((0, 20), "Size:"+str(img.size[0])+" x "+str(img.size[1]),font=font,fill=(0,0,0,255))
    img.save(os.path.join(savepath,dataset+"_"+name[0:13]+'_LR.png'),"PNG")


opt = parser.parse_args()
model=torch.load(opt.model_name)
print(opt)



datalist=['Set5']#,'Set14','BSD100']
for dl in datalist:
    if os.path.isdir(dl) is False:
        os.makedirs(dl)
    savepath=os.path.join(os.getcwd(),dl)
    datapath=os.path.join(os.getcwd(),'dataset/data/'+dl+'/image_SRF_'+str(opt.upscale_factor))
    name='img_000_SRF_2_LR.png'
    if opt.upscale_factor is not 2:
        name=name.replace("2",str(opt.upscale_factor))
    if dl is "BSD100":
        matrix=[0]*4
        # 0: BICUBIC PSNR 1: SR PSNR 2: BICUBIC SSIM 3: SR SSIM
        for i in range(1,101):
            inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)

        if opt.visual <= 2:
            print('BSD100 average BICUBIC PSNR: ',matrix[0]/100)
            print('BSD100 average OURS PSNR: ',matrix[1]/100)
            print('BSD100 average BICUBIC SSIM: ',matrix[2]/100)
            print('BSD100 average OURS SSIM: ',matrix[3]/100)
    elif dl is "Set5":
        matrix=[0]*4
        for i in range(1,6):
            inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)

        if opt.visual <= 2:
            print('Set5 average BICUBIC PSNR: ',matrix[0]/5)
            print('Set5 average OURS PSNR: ',matrix[1]/5)
            print('Set5 average BICUBIC SSIM: ',matrix[2]/5)
            print('Set5 average OURS SSIM: ',matrix[3]/5)
    elif dl is "Set14":
        matrix=[0]*4
        for i in range(1,15):
            inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)
        if opt.visual <= 2:
            print('Set14 average BICUBIC PSNR: ',matrix[0]/14)
            print('Set14 average OURS PSNR: ',matrix[1]/14)
            print('Set14 average BICUBIC SSIM: ',matrix[2]/14)
            print('Set14 average OURS SSIM: ',matrix[3]/14)
    else:
        print("Finish!")
