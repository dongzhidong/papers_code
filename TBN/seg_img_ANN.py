#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2020/10/24 13:32
# @Author  : dongdong
# @File    : seg_img_ANN.py
# @Software: PyCharm
# 使用CNN 对segdata 和 rawdata 进行分类
from paramters import *
import tools as tl
import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import average_precision_score,accuracy_score

# read data and set paramters
image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
Seg_img=sio.loadmat(os.path.join(PaviaU_path,'Pavia_seg.mat'))['results']
test_rate=0.97
channel_show=100

# data preprocess
# segdata=tl.my_pca(Seg_img)  #对segdata pca
segdata=Seg_img
segdata=tl.preprocess(segdata) # 归一化
segdata=segdata.reshape([segdata.shape[0]*segdata.shape[1],segdata.shape[2]]) # 列向量

# imgdata=tl.my_pca(image)  #对image pca
imgdata=image
imgdata=tl.preprocess(imgdata) # 归一化
imgdata=imgdata.reshape([imgdata.shape[0]*imgdata.shape[1],imgdata.shape[2]]) # 列向量
labels=labels.reshape([labels.shape[0]*labels.shape[1],1])

# pavia oa 0.96
data=np.concatenate((segdata,imgdata),axis=1)
save_name='rawseg'
# data=segdata
# save_name='seg'
# data=imgdata
# save_name='raw'

segdata=data
remain_index=np.argwhere(labels!=0)[:,0]
# 对可用元素的 索引划分 训练集 测试集
trainX_ind,testX_ind,trainY,testY=train_test_split(remain_index,labels[remain_index],train_size=50*9,random_state=2020)
print(trainX_ind.shape,testX_ind.shape)
class AnnNet(nn.Module):
    def __init__(self, channels,n_classes, dropout=False, **kwargs):
        super(AnnNet,self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.fc1=nn.Linear(channels,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,64)
        self.fc4 = nn.Linear(64, n_classes)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x
    
class ComNet(nn.Module):
    def __init__(self, img_channels,seg_channels,n_classes, dropout=False, **kwargs):
        super(ComNet,self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.fc11 = nn.Linear(img_channels, 2048)
        self.fc12 = nn.Linear(2048, 4096)
        self.fc13 = nn.Linear(4096, 2048)
        self.fc14 = nn.Linear(2048, n_classes)
        self.fc21 = nn.Linear(seg_channels, 2048)
        self.fc22 = nn.Linear(2048, 4096)
        self.fc23 = nn.Linear(4096, 2048)
        self.fc24 = nn.Linear(2048, n_classes)
        self.fc1=nn.Linear(n_classes*2,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,n_classes)

    def forward(self,img_x,seg_x):
        img_x = F.relu(self.fc11(img_x))
        if self.use_dropout:
            img_x = self.dropout(img_x)
        img_x = F.relu(self.fc12(img_x))
        if self.use_dropout:
            img_x = self.dropout(img_x)
        img_x = F.relu(self.fc13(img_x))
        if self.use_dropout:
            img_x = self.dropout(img_x)
        img_x = self.fc14(img_x)

        seg_x = F.relu(self.fc21(seg_x))
        if self.use_dropout:
            seg_x = self.dropout(seg_x)
        seg_x = F.relu(self.fc22(seg_x))
        if self.use_dropout:
            seg_x = self.dropout(seg_x)
        seg_x = F.relu(self.fc23(seg_x))
        if self.use_dropout:
            seg_x = self.dropout(seg_x)
        seg_x = self.fc24(seg_x)

        x=F.relu(torch.cat((img_x,seg_x),dim=1))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))

        return x




# 训练过程
# 超参数
def train():
    def val():
        # print(' val...\r',end='')
        val_ind, _, valy, _ = train_test_split(testX_ind, testY, train_size=0.05)
        # img_x = torch.tensor(imgdata[val_ind], dtype=torch.float32, requires_grad=False).cuda(device)
        seg_x = torch.tensor(segdata[val_ind], dtype=torch.float32, requires_grad=False).cuda(device)
        target = valy - 1
        target = target.T[0]
        annNet.eval()
        pred = annNet(seg_x).cpu().detach().numpy()
        pred = np.array([np.argwhere(pred[x, :] == max(pred[x, :]))[0][0] for x in range(len(pred))])
        oa = accuracy_score(target, pred)
        return oa
    def test():
        print('\ntest...\r', end='')
        iters = len(testX_ind) // batch
        oa_end = 0.0
        annNet.eval()
        for i in range(iters):
            print('\rtest {}/{}...'.format(i, iters), end='')
            b = ((i + 1) * batch) if (i + 1) * batch < len(testX_ind) else -1
            test_ind = testX_ind[i * batch:b]
            # img_x = torch.tensor(imgdata[test_ind], dtype=torch.float32, requires_grad=False).cuda(device)
            seg_x = torch.tensor(segdata[test_ind], dtype=torch.float32, requires_grad=False).cuda(device)
            target = labels[test_ind] - 1
            target = target.T[0]

            pred = annNet(seg_x).cpu().detach().numpy()
            pred = np.array([np.argwhere(pred[x, :] == max(pred[x, :]))[0][0] for x in range(len(pred))])
            oa = accuracy_score(target, pred)

            oa_end += oa

        return oa_end / iters

    epochs=1000
    lr=0.1
    batch=256

    n_classes=np.unique(labels[remain_index]).shape[0]
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    torch.cuda.empty_cache()
    annNet=AnnNet(data.shape[1],n_classes)
    annNet.to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.Adam(annNet.parameters(), lr = lr)
    optimizer = optim.SGD(annNet.parameters(), lr = lr)
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=200,verbose=True,min_lr=1.0e-3)
    print('init paramters:\nepochs:{} lr:{} batch:{} '.format(epochs,lr,batch))
    val_max_oa=0.0
    for epoch in range(epochs):
        annNet.train()
        iters=len(trainX_ind)//batch;
        np.random.shuffle(trainX_ind)
        loss_mean=0.0
        for i in range(iters):
            train_ind=trainX_ind[i*batch:(i+1)*batch if (i+1)*batch<len(trainX_ind) else i*batch:]
            # img_x=torch.tensor(imgdata[train_ind],dtype=torch.float32).cuda(device)
            seg_x=torch.tensor(segdata[train_ind],dtype=torch.float32).cuda(device)
            target=torch.tensor(labels[train_ind]-1,dtype=torch.int64).cuda(device)

            optimizer.zero_grad()
            output=annNet(seg_x)
            loss = criterion(output, target.T[0])
            loss_mean+=loss
            loss.backward()
            optimizer.step()
            print('\repoch:{}/{} iter:{}/{} loss:{}'.format(epoch+1,epochs,i,iters,loss),end='')
        val_oa=val()
        lr_scheduler.step(loss_mean/iters)
        print(' val OA:{}'.format(val_oa),' lr:',optimizer.param_groups[0]['lr'])
        if val_max_oa<val_oa:
            torch.save(annNet.state_dict(),os.path.join(model_path,'ann_best_model.pkl'))
            val_max_oa=val_oa
            print('Saving best model...',val_max_oa)

        # if (epoch+1)%25==0:
        #     print('epoch:{}/{} test OA:{}\n'.format(epoch+1, epochs, test()))

    testOA=test()
    print('last test OA:',testOA)
    # 保存
    model_name='annNet_state_dict_'+save_name+'_NN_{}.pkl'.format(testOA)
    torch.save(annNet.state_dict(),os.path.join(model_path,model_name))
    # 加载
    annNet=AnnNet(data.shape[1],n_classes)
    annNet.load_state_dict(torch.load(os.path.join(model_path,'ann_best_model.pkl')))
    annNet.cuda()
    testOA=test()
    print('ann best model test OA:',testOA)
    model_name='annNet_state_dict_'+save_name+'_NN_{}.pkl'.format(testOA)
    torch.save(annNet.state_dict(),os.path.join(model_path,model_name))
    return testOA

if __name__ == '__main__':
    testOA=[]
    run_nums=10
    for run_id in range(run_nums):
        print('run_id:{}/{}-----------------'.format(run_id,run_nums))
        testOA.append(train())
    print(testOA)
    print('end')












