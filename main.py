from __future__ import print_function
import argparse
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import datetime,random,os,csv
from torch.nn.modules.module import _addindent
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--data', type=str, default='BSDS300',required=False, help="train data path")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true' ,help='use cuda?')
parser.add_argument('--threads', type=int, default=12, help='number of threads for data loader to use')
parser.add_argument('--model', type=int, default='1', help='choose a model')

opt = parser.parse_args()
name=opt.data
if opt.model is 1:
    from model import Net
    name+=' default_model_non_depthwise'
elif opt.model is 2:
    from model_dw import Net
    name+=' default_model_depthwise'
elif opt.model is 3:  
    from mymodel import Net
    name+=' custom_model_non_depthwise'
elif opt.model is 4:  
    from mymodel_dw import Net
    name+=' custom_model_depthwise'
else:
    from model import Net
    name+=' default_model_non_depthwise'
#parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
#opt = parser.parse_args()
_time="result/"+str(datetime.datetime.now())[:10]+"_"+str(datetime.datetime.now())[11:-7]
os.makedirs(_time)
f = open(os.path.join(os.path.join(os.getcwd(),_time),name+".txt"), 'w')
f.write(str(opt))

print(opt)
fc = open(os.path.join(os.path.join(os.getcwd(),_time),name+".csv"), 'w', encoding='utf-8', newline='')
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
train_set = get_training_set(opt.upscale_factor,opt.data)
test_set = get_test_set(opt.upscale_factor,opt.data)
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

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

print(torch_summarize(model))
print('\n num of parameters ',params)
f.write(torch_summarize(model))
wr.writerow([idx,'Num of parameters',params])
idx+=1
f.write('Num of parameters '+str(params))
def train(epoch):
    epoch_loss = 0
    global idx
    global f
    global wr
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
        wr.writerow([idx,'Loss',epoch,iteration,loss.data[0]])
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
    global f
    global wr
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
