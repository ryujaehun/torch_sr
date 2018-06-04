#!/usr/bin/python3
from __future__ import print_function
import argparse
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.data import  get_test_set
import torch.backends.cudnn as cudnn
from utils.dataset2 import DatasetFromHdf5
import datetime,random,os
from utils.logger import Logger,to_np
from torchvision.transforms import ToTensor
from PIL import Image,ImageFont, ImageDraw

import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
from pandas import DataFrame

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor','-u', type=int,default=2, required=False, help="super resolution upscale factor")
parser.add_argument('--data', type=str,default='SRCNN',required=False, help="train data path")
parser.add_argument('--batchSize','-b', type=int, default=128, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs','-n', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true' ,help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--model','-m', type=int, default='1', help='name of log file name')
parser.add_argument('--dict', type=bool, default=False, help='Saveing option dict')
parser.add_argument('--save_interval','-s', type=int, default=20, help='saveing interval')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight-decay", "-wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--step", type=int, default=7, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--clip", type=float, default=0.01, help="Clipping Gradients, Default=0.01")
opt = parser.parse_args()
name=''
model_out_path=""
if opt.model is 1:
    from net.model1 import Net
    name+='model_1_'
elif opt.model is 2:
    from net.model2 import Net
    name+='model_2_'
elif opt.model is 3:
    from net.model3 import Net
    name+='model_3_'
elif opt.model is 4:
    from net.model4 import Net
    name+='model_4_'
else:
    print("illigel model!!\n")
    exit()
name+=str(opt.upscale_factor)
_time="result/"+name+'/'+str(datetime.datetime.now())[:10]+"_"+str(datetime.datetime.now())[11:-7]
os.makedirs(_time)
logger = Logger(_time)
print(opt)
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed(random.randint(1,10000))
print('===> Loading datasets')

if opt.upscale_factor==2:
    train_set = DatasetFromHdf5("dataset/train_2_96_28.h5")
elif opt.upscale_factor==3:
    train_set = DatasetFromHdf5("dataset/train_3_96_28.h5")
elif opt.upscale_factor==4:
    train_set = DatasetFromHdf5("dataset/train_4_96_28.h5")
training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)
test_set = get_test_set(opt.upscale_factor,opt.data)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor)
criterion = nn.MSELoss()

if cuda:
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("\nNum of parameters",params)
logger.scalar_summary('paramter',params,0)
def train(epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logger.scalar_summary('learning rate',lr,epoch)
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.data[0]
        loss.backward()
        clip = opt.clip / lr
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        logger.scalar_summary('loss',loss.data[0], iteration+epoch*len(training_data_loader)+1)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    logger.scalar_summary('total loss', epoch_loss / len(training_data_loader), epoch+1)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), epoch+1)
            logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)

def test(epoch):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    logger.scalar_summary('PSNR',avg_psnr / len(testing_data_loader), epoch)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if opt.lr<0.0005:
        return opt.lr
    lr = opt.lr * (0.6 ** (epoch  // opt.step))
    return lr



def checkpoint(epoch,_dict=False):
    global model_out_path
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path=os.path.join(os.path.join(os.getcwd(),_time),model_out_path)
    torch.save(model, model_out_path)
    print("Checkpoint pth saved to {}".format(model_out_path))
    if _dict is True:
        model_out_dict_path = "model_dict_epoch_{}".format(epoch)
        model_out_dict_path=os.path.join(os.path.join(os.getcwd(),_time),model_out_dict_path)
        torch.save(model.state_dict(), model_out_dict_path)
        print("Checkpoint dict saved to {}".format(model_out_dict_path))
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def infer():
    global model
    index=['avg_psnr_predicted','avg_psnr_bicubic','avg_elapsed_time']
    df=DataFrame(index=index)
    dataset=['Set5_mat','Set14_mat','Urban100_mat','B100_mat']
    for i_list in dataset:
        image_list = glob.glob(i_list+"/*.*")
        count=0
        avg_psnr_predicted = 0.0
        avg_psnr_bicubic = 0.0
        avg_elapsed_time = 0.0
        for image_name in image_list:
            if not 'x'+str(opt.upscale_factor) in image_name:
                count+=1
                continue
            im_gt = sio.loadmat(image_name)['im_gt_ycbcr']
            im_b = sio.loadmat(image_name)['im_b']
            im_l = sio.loadmat(image_name)['im_l']
            im_gt_y = im_gt[:,:,0].astype(float)
            im_b_y = im_b[:,:,0].astype(float)
            im_l_y = im_l[:,:,0].astype(float)

            psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.upscale_factor)
            avg_psnr_bicubic += psnr_bicubic
            im_input = im_l_y/255.
            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            HR_result= model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            HR_result = HR_result.cpu()


            im_h_y = HR_result.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y*255.
            im_h_y[im_h_y<0] = 0
            im_h_y[im_h_y>255.] = 255.
            im_h_y = im_h_y[0,:,:]
            psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.upscale_factor)
            avg_psnr_predicted += psnr_predicted
            result=Image.fromarray(ycbcr2rgb(np.stack((im_h_y,im_b[:,:,1],im_b[:,:,2]),axis=2)))
            result_b=Image.fromarray(ycbcr2rgb(im_b))
            font = ImageFont.truetype("arial.ttf", 12)
            draw = ImageDraw.Draw(result_b)
            #draw.rectangle([0,0,120,20], fill=(255,255,255,255))
            draw.text((0, 0), "BICUBIC",font=font,fill=(0,0,0,255))
            draw.text((0, 24), "PSNR:"+str(psnr_bicubic),font=font,fill=(0,0,0,255))
            if not os.path.exists(os.path.join(model_out_path[:-19],image_name.split('/')[0])):
                os.makedirs(os.path.join(model_out_path[:-19],image_name.split('/')[0]))
            result_b.save(os.path.join(model_out_path[:-19],image_name[:-4]+'_bicubic.png'),"PNG")

            draw = ImageDraw.Draw(result)
            #draw.rectangle([0,0,120,20], fill=(255,255,255,255))
            draw.text((0, 0), "Super Resolution",font=font,fill=(0,0,0,255))
            draw.text((0, 24), "PSNR:"+str(psnr_predicted),font=font,fill=(0,0,0,255))
            result.save(os.path.join(model_out_path[:-19],image_name[:-4]+'_superresolution.png'),"PNG")


        print("Scale=", opt.upscale_factor)
        print("Dataset=",  i_list[:-4])
        print("PSNR_predicted=", avg_psnr_predicted/(len(image_list)-count))
        print("PSNR_bicubic=", avg_psnr_bicubic/(len(image_list)-count))
        print("It takes average {}s for processing".format(avg_elapsed_time/(len(image_list)-count)))
        df[i_list[:-4]]=[avg_psnr_predicted/(len(image_list)-count),avg_psnr_bicubic/(len(image_list)-count)
                        ,avg_elapsed_time/(len(image_list)-count)]
    df.to_csv(os.path.join(model_out_path[:-19],'psnr_result.csv'), encoding='utf-8')

if __name__ == "__main__":
    for epoch in range(1, opt.nEpochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        if epoch%opt.save_interval==0:
            checkpoint(epoch,_dict=opt.dict)
    infer()
