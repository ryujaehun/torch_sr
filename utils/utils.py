
def modify(net,alpha=64,beta=64,gamma=32):
    """
    네트워크와 하이퍼 파라미터를 입력받아
    네트워크를 하이퍼 파라미터에 맞게 변형한다.
    """
    net.conv1 = nn.Conv2d(1, alpha, (5, 5), (1, 1), (2, 2))
    net.conv2=nn.Conv2d(alpha, beta, (3, 3), (1, 1), (1, 1))
    net.conv3=nn.Conv2d(beta, gamma, (3, 3), (1, 1), (1, 1))
    net.conv4 = nn.Conv2d(gamma, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))

def PSNR(net):
    '''
    네트워크 를 받아서 psnr 값을 구하여 반환한다.
    테스트 배치 만큼 수행 평균
    '''
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target=target.cuda()
        prediction = net(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    return avg_psnr/len(testing_data_loader)

def _train(net,epoch=50,lr):
    '''
    네트워크와 에폭,학습률을 입력 받아 그에 맞게 학습시킨다.
    '''
    #net=nn.DataParallel(net) 이렇게 하는게 더 느리다 % time으로 확인
    optimizer = optim.Adam(net.parameters(), lr)
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        loss = criterion(net(input), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    if epoch%10 is 0:
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def rank(_list):
    '''
    _list를 입력받아 계산한뒤 pruning 할 (conv,weight)의 위치를 tuple로 반환한다.
    pruning 할 layer를 고른다.
    기준은 L2와 PSNR 을 기준으로 rank를 만든 뒤 합하여 결정한다.
    '''
    df=DataFrame(_list,columns=['conv','weight','psnr','L2'])
    df['rank']=df['psnr'].rank( ascending=False,method='max')+df['L2'].rank(method='max')
    for idx,i in enumerate(df["L2"]):
        if i == min(df["L2"]):
            print(df['conv'][idx],"th conv's ",df['weight'][idx],"'s layer will pruning")
            return(df['conv'][idx],df['weight'][idx])
def _pruning(_model,_weight,index,epoch=50):
    """
    _model: 모델을 받는다.
    _weight:weight값을 받는다.
    index: pruning을 시행할 convolution 의 index 값을 받는다.
    epoch: pruning후 retrain 할 횟수
    """
    keys_list=list(_weight.keys())
    temp=[]
    for i in range(0,len(keys_list)-2,2):
        temp.append(len(_weight[keys_list[i]]))
    alpha,beta,gamma=temp

    print('temp',temp,'alpha',alpha,'beta',beta,'gamma',gamma)
    i=0
    j=index[1]

    if index[0] is 1:
        i=0
        alpha-=1
        modify(_train,alpha,beta,gamma)
    elif index[0] is 2:
        i=2
        beta-=1
        modify(_train,alpha,beta,gamma)
    elif index[0] is 3:
        i=4
        gamma-=1
        modify(_train,alpha,beta,gamma)
    if j>len(_weight[keys_list[i]]):
        print("illegal pruning")
        return
    weight_matrix=_weight[keys_list[i]]
    bias_matrix=_weight[keys_list[i+1]]
    if index[1] is 0:
        _weight[keys_list[i]]=weight_matrix[1:len(_weight[keys_list[i]])]
        _weight[keys_list[i+1]]=bias_matrix[1:len(_weight[keys_list[i]])+1]
    elif j is len(model[keys_list[i]])-1:
        _weight[keys_list[i]]=weight_matrix[0:len(_weight[keys_list[i]])-1]
        _weight[keys_list[i+1]]=bias_matrix[0:len(_weight[keys_list[i]])]
    else:
        _weight[keys_list[i]]=torch.cat((weight_matrix[0:j],weight_matrix[j+1:len(_weight[keys_list[i]])]))
        _weight[keys_list[i+1]]=torch.cat((bias_matrix[0:j],bias_matrix[j+1:len(_weight[keys_list[i]])+1]))
    if i is 0:
        _weight[keys_list[i+2]].resize_(alpha,beta,3,3)
    elif i is 2:
        _weight[keys_list[i+2]].resize_(beta,gamma,3,3)
    elif i is 4:
        _weight[keys_list[i+2]].resize_(gamma,upscale_factor ** 2,3,3)
    _model=_model.cuda()
    for k in range(0,epoch):
        _train(_model,k)
    prnr=PSNR(_model)
    print('conv:',i,' num:',j,'is pruning psnr:',prnr)
def cal_pruning(_model,_weight_soruce,retrain=True,epoch=200):
    _weight=deepcopy(_weight_soruce)
    print("===> Starting Calculate Pruning")
    keys_list=list(_weight.keys())
    temp=[]
    for i in range(0,len(keys_list)-2,2):
        temp.append(len(_weight[keys_list[i]]))
    alpha,beta,gamma=temp
    print(alpha,beta,gamma)
    psnr_list=[]
    for i in range(0,len(keys_list)-2,2):
        if i is 0:
            alpha-=1
            modify(_model,alpha,beta,gamma)
        elif i is 2:
            alpha+=1
            beta-=1
            modify(_model,alpha,beta,gamma)
        elif i is 4:
            beta+=1
            gamma-=1
            modify(_model,alpha,beta,gamma)
            _weight=deepcopy(_weight_soruce)
        for j in range(len(_weight[keys_list[i]])):
            _weight=deepcopy(_weight_soruce)
            weight_matrix=_weight[keys_list[i]]
            bias_matrix=_weight[keys_list[i+1]]
            temp_weight=0
            if j is 0:
                temp_weight=weight_matrix[0].abs().sum()
                _weight[keys_list[i]]=weight_matrix[1:len(_weight[keys_list[i]])]
                _weight[keys_list[i+1]]=bias_matrix[1:len(_weight[keys_list[i]])+1]
            elif j is len(_weight[keys_list[i]])-1:
                temp_weight=weight_matrix[len(_weight[keys_list[i]])-1].abs().sum()
                _weight[keys_list[i]]=weight_matrix[0:len(_weight[keys_list[i]])-1]
                _weight[keys_list[i+1]]=bias_matrix[0:len(_weight[keys_list[i]])]
            else:
                temp_weight=weight_matrix[j].abs().sum()
                _weight[keys_list[i]]=torch.cat((weight_matrix[0:j],weight_matrix[j+1:len(_weight[keys_list[i]])]))
                _weight[keys_list[i+1]]=torch.cat((bias_matrix[0:j],bias_matrix[j+1:len(_weight[keys_list[i]])+1]))
            if i is 0:
                _weight[keys_list[i+2]].resize_(alpha,beta,3,3)
            elif i is 2:
                _weight[keys_list[i+2]].resize_(beta,gamma,3,3)
            elif i is 4:
                _weight[keys_list[i+2]].resize_(gamma,upscale_factor ** 2,3,3)
            _model.load_state_dict(_weight)
            _model=_model.cuda()
            if retrain is True:
                for k in range(0,epoch):
                    _train(_model,k)
            prnr=PSNR(_model)
            print('conv:',i,' num:',j,' psnr:',prnr,' size:',temp_weight)
            psnr_list.append(tuple([i,j,prnr,temp_weight]))
    return psnr_list
