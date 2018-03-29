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
from utils.data import get_training_set, get_test_set
import datetime,random,os
from utils.logger import Logger,to_np
from utils.metric import psnr,ssim
from PIL import Image,ImageFont, ImageDraw
from torchvision.transforms import ToTensor
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor', type=int,default=2, required=False, help="super resolution upscale factor")
parser.add_argument('--data', type=str,default='SRCNN',required=False, help="train data path")
parser.add_argument('--batchSize','-b', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs','-n', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true' ,help='use cuda?')
parser.add_argument('--threads', type=int, default=11, help='number of threads for data loader to use')
parser.add_argument('--model', type=int, default='1', help='name of log file name')
parser.add_argument('--dict', type=bool, default=False, help='Saveing option dict')
parser.add_argument('--save_interval','-s', type=int, default='50', help='saveing interval')
opt = parser.parse_args()
name=''

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
elif opt.model is 5:
    from net.model5 import Net
    name+='model_5_'
elif opt.model is 6:
    from net.model6 import Net
    name+='model_6_'
elif opt.model is 7:
    from net.model7 import Net
    name+='model_7_'
elif opt.model is 8:
    from net.model8 import Net
    name+='model_8_'
elif opt.model is 9:
    from net.model9 import Net
    name+='model_9_'
elif opt.model is 10:
    from net.model11 import Net
    name+='model_10_'
elif opt.model is 11:
    from net.model10 import Net
    name+='model_11_'
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
torch.manual_seed(random.randint(1,1000))
if cuda:
    torch.cuda.manual_seed(random.randint(1,1000))
print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor,opt.data)
test_set = get_test_set(opt.upscale_factor,opt.data)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor)
criterion = nn.MSELoss()

if cuda:
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("\nNum of parameters",params)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.data[0]
        loss.backward()
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
    lr = opt.lr * (0.9 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.scalar_summary('learning rate',lr,epoch)

def checkpoint(epoch,_dict=False):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path=os.path.join(os.path.join(os.getcwd(),_time),model_out_path)
    torch.save(model, model_out_path)
    print("Checkpoint pth saved to {}".format(model_out_path))
    if _dict is True:
        model_out_dict_path = "model_dict_epoch_{}".format(epoch)
        model_out_dict_path=os.path.join(os.path.join(os.getcwd(),_time),model_out_dict_path)
        torch.save(model.state_dict(), model_out_dict_path)
        print("Checkpoint dict saved to {}".format(model_out_dict_path))

def inference(epoch,savepath,datapath,name,dataset):
    global model
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
    global matrix
    img=img.convert('RGB')
    if dataset is "Set14" and epoch is 3:
        img_bicubic=img_bicubic.convert('RGB')
        img_hr=img_hr.convert('RGB')
    matrix=[sum(x) for x in zip(matrix, [psnr(img_bicubic,img_hr),psnr(out_img,img_hr),ssim(img_bicubic,img_hr),ssim(out_img,img_hr)])]
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

if __name__ == "__main__":
    for epoch in range(1, opt.nEpochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        if epoch%opt.save_interval==0:
            checkpoint(epoch,_dict=opt.dict)
    datalist=['Set5','Set14','BSD100']
    for dl in datalist:
        savepath=os.path.join(os.path.join(os.getcwd(),_time),dl)
        datapath=os.path.join(os.getcwd(),'dataset/data/'+dl+'/image_SRF_'+str(opt.upscale_factor))
        if os.path.isdir(savepath) is False:
            os.makedirs(savepath)
        name='img_000_SRF_2_LR.png'
        if opt.upscale_factor is not 2:
            name=name.replace("2",str(opt.upscale_factor))
        if dl is "BSD100":
            matrix=[0]*4
            # 0: BICUBIC PSNR 1: SR PSNR 2: BICUBIC SSIM 3: SR SSIM
            for i in range(1,101):
                inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)
            print('BSD100 average BICUBIC PSNR: ',matrix[0]/100)
            print('BSD100 average OURS PSNR: ',matrix[1]/100)
            print('BSD100 average BICUBIC SSIM: ',matrix[2]/100)
            print('BSD100 average OURS SSIM: ',matrix[3]/100)
        elif dl is "Set5":
            #name=name.replace("png",'bmp')
            matrix=[0]*4
            for i in range(1,6):
                inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)
            print('Set5 average BICUBIC PSNR: ',matrix[0]/5)
            print('Set5 average OURS PSNR: ',matrix[1]/5)
            print('Set5 average BICUBIC SSIM: ',matrix[2]/5)
            print('Set5 average OURS SSIM: ',matrix[3]/5)
        elif dl is "Set14":
            matrix=[0]*4
            #name=name.replace("png",'bmp')
            for i in range(1,15):
                inference(epoch=i,savepath=savepath,datapath=datapath,name=name.replace("000",str(i).rjust(3, '0')),dataset=dl)
            print('Set14 average BICUBIC PSNR: ',matrix[0]/14)
            print('Set14 average OURS PSNR: ',matrix[1]/14)
            print('Set14 average BICUBIC SSIM: ',matrix[2]/14)
            print('Set14 average OURS SSIM: ',matrix[3]/14)
        else:
            print("Finish!")
