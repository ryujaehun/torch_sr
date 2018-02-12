from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mymodel2 import Net
from data import get_training_set, get_test_set
import datetime,random,os,csv
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--data', type=str, required=False, help="train data path")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', required=True,help='use cuda?')
parser.add_argument('--threads', type=int, default=12, help='number of threads for data loader to use')
#parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
_time="result/"+str(datetime.datetime.now())[:10]+"_"+str(datetime.datetime.now())[11:-7]
os.makedirs(_time)
f = open(os.path.join(os.path.join(os.getcwd(),_time),"logging.txt"), 'w')
f.write(str(opt))

print(opt)
fc = open(os.path.join(os.path.join(os.getcwd(),_time),"logging.csv"), 'w', encoding='utf-8', newline='')
wr = csv.writer(fc)
idx=0
wr.writerow([idx,'upscale_factor',opt.upscale_factor])
idx+=1
wr.writerow([idx,'batchSize',opt.batchSize])
idx+=1
wr.writerow([idx,'testBatchSize',opt.testBatchSize])
idx+=1
wr.writerow([idx,'Epoch',opt.nEpochs])
idx+=1
wr.writerow([idx,'learningRate',opt.lr])
idx+=1
wr.writerow([idx,'threads',opt.threads])
idx+=1

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(random.randint(1,1000))
if cuda:
    torch.cuda.manual_seed(random.randint(1,1000))
f.write('===> Loading datasets')
print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
f.write('===> Building model')
print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor)
criterion = nn.MSELoss()

if cuda:
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
       # criterion=nn.DataParallel(criterion)

    if torch.cuda.is_available():
    
        model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    global idx
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
        f.write("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        wr.writerow([idx,'Loss'])
        idx+=1
    wr.writerow([idx,'Avg_ioss',epoch,epoch_loss / len(training_data_loader)])
    idx+=1
    f.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


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
    global idx
    wr.writerow([idx,'PSNR',epoch,avg_psnr / len(testing_data_loader)])
    idx+=1
    f.write("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path=os.path.join(os.path.join(os.getcwd(),_time),model_out_path)
    torch.save(model, model_out_path)
    f.write("Checkpoint saved to {}".format(model_out_path))
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test(epoch)
    checkpoint(epoch)
fc.close()
f.close()
