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
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--cuda', action='store_true' ,help='use cuda?')
parser.add_argument('--threads', type=int, default=12, help='number of threads for data loader to use')
parser.add_argument('--model', type=int, default='1', help='name of log file name')
parser.add_argument('--weight', type=str, required=True, help='name of log file name')

opt = parser.parse_args()
if opt.model is 1:
    from model import Net
    name='default model non depthwise'
elif opt.model is 2:
    from model_dw import Net
    name='default model depthwise'
elif opt.model is 3:  
    from mymodel import Net
    name='custom model non depthwise'
elif opt.model is 4:  
    from mymodel_dw import Net
    name='custom model depthwise'
else:
    from model import Net
    name='default model non depthwise'
#parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
#opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(random.randint(1,1000))
if cuda:
    torch.cuda.manual_seed(random.randint(1,1000))
print('===> Loading datasets')
test_set = get_test_set(opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
print('===> Building model')
model =torch.load(opt.weight)
criterion = nn.MSELoss()

if cuda:
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
       # criterion=nn.DataParallel(criterion)

    if torch.cuda.is_available():
    
        model = model.cuda()
    criterion = criterion.cuda()


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
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))



for epoch in range(1,10 ):
    test(epoch)
