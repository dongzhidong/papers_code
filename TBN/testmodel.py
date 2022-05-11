#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2020/11/1 10:27
# @Author  : dongdong
# @File    : segCNN.py
# @Software: PyCharm

from paramters import *
import tools
import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,accuracy_score
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from sklearn import  metrics
def load_data(dateset_name,load_seg=True,**kwargs):
    """ 加载数据，加载train_size n_classes参数"""
    if dateset_name=='IndianPines':
        image = sio.loadmat(os.path.join(IndianPine_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(IndianPine_path, 'labels.mat'))['labels'].reshape(145, 145).T
        kwargs.setdefault('train_size',0.1)
        kwargs.setdefault('n_classes', 16)
        if load_seg:
            seg_img = sio.loadmat(os.path.join(IndianPine_path, 'Indian_seg.mat'))['seg_labels']
    elif dateset_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        kwargs.setdefault('train_size', 0.03)
        kwargs.setdefault('n_classes', 9)
        if load_seg:
            seg_img=sio.loadmat(os.path.join(PaviaU_path,'Pavia_seg.mat'))['results']

    if load_seg:
        image=np.array(image,dtype=np.float)
        seg_img=np.array(seg_img,dtype=np.float)
        labels = np.array(labels, dtype=np.int)
        # 归一化
        image = tools.preprocess(tools.my_pca(image))
        seg_img = tools.preprocess(tools.my_pca(seg_img))
        kwargs.setdefault('w', image.shape[0])
        kwargs.setdefault('h', image.shape[1])
        kwargs.setdefault('b', image.shape[2])
        kwargs.setdefault('seg_w', seg_img.shape[0])
        kwargs.setdefault('seg_h', seg_img.shape[1])
        kwargs.setdefault('seg_b', seg_img.shape[2])
        kwargs.setdefault('input_channels',seg_img.shape[2])
        return image,labels,seg_img,kwargs
    return image,labels,kwargs

class mydataset(Dataset):
    def __init__(self,img,labels,seg_img,data_ind,**kwargs):
        self.name=kwargs.get('dataset_name')
        self.patch = kwargs.get('patch_size')
        # padding 0
        # self.img= np.lib.pad(img,((self.patch//2,self.patch//2),(self.patch//2,self.patch//2),(0,0)))
        self.seg_img=0
        self.labels=labels
        self.seg_img=np.lib.pad(seg_img,((self.patch//2,self.patch//2),(self.patch//2,self.patch//2),(0,0)))
        self.data_ind=data_ind
        self.kwargs=kwargs

    def __getitem__(self, item):
        x,y=self.data_ind[item,:]
        # 得到box 注意padding后坐标变化
        # sub_img=self.img[x:x+self.patch,y:y+self.patch]
        sub_img=0
        sub_labels=self.labels[x,y]
        sub_seg=self.seg_img[x:x+self.patch,y:y+self.patch]
        return 0,sub_labels,sub_seg
        # return sub_img,sub_labels,sub_seg
    def __len__(self):
        return len(self.data_ind)

class myCNN(nn.Module):
    def __init__(self,input_channels,n_classes,patch_size):
        super(myCNN,self).__init__()
        self.input_channels=input_channels
        self.n_classes=n_classes
        self.patch_size=patch_size
        self.conv1=nn.Conv3d(1,64,(3,3,3),padding=(0,1,1))
        self.conv2=nn.Conv3d(64,32,(3,3,3),padding=(0,1,1))
        self.conv3=nn.Conv3d(32,16,(3,3,3),padding=(0,1,1))
        self.fc_size=self.get_fc_size()
        self.fc1=nn.Linear(self.fc_size,64)
        self.fc2=nn.Linear(64,n_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view(-1,self.fc_size)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x


        pass
    def get_fc_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1,self.input_channels,self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t*c*w*h

def val(val_loader,cnnNet,**kwargs):
    iter = 0
    device=kwargs.get('device')
    cnnNet.eval()
    preds=[]
    labels=[]
    for sub_img, sub_labels, sub_seg in val_loader:
        iter = iter + 1
        # sub_img = np.transpose(sub_img, (0, 3, 1, 2))
        sub_seg = np.transpose(sub_seg, (0, 3, 1, 2))
        x = torch.as_tensor(sub_seg.to(device), dtype=torch.float32).to(device)

        x = torch.unsqueeze(x, 1)
        # sub_labels = torch.unsqueeze(sub_labels, 0)
        output = cnnNet(x)
        pred=output.cpu().detach().numpy()
        pred=list(np.argmax(pred,axis=1))
        preds.extend(pred)
        labels.extend(list(sub_labels.cpu().detach().numpy()-1))

    oa=metrics.accuracy_score(labels,preds)
    aa=metrics.precision_score(labels,preds,average='micro')
    kp = metrics.cohen_kappa_score(labels, preds)
    return oa,aa,kp

def test(test_loader,cnnNet,**kwargs):
    iter = 0
    device=kwargs.get('device')
    cnnNet.eval()
    preds=[]
    labels=[]
    for sub_img, sub_labels, sub_seg in test_loader:
        iter = iter + 1
        # sub_img = np.transpose(sub_img, (0, 3, 1, 2))
        sub_seg = np.transpose(sub_seg, (0, 3, 1, 2))
        x = torch.as_tensor(sub_seg.to(device), dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, 1)
        # sub_labels = torch.unsqueeze(sub_labels, 0)
        output = cnnNet(x)
        pred=output.cpu().detach().numpy()
        pred=list(np.argmax(pred,axis=1))
        preds.extend(pred)
        labels.extend(list(sub_labels.cpu().detach().numpy()-1))
    oa=metrics.accuracy_score(labels,preds)
    aa=metrics.precision_score(labels,preds,average='micro')
    kp = metrics.cohen_kappa_score(labels, preds)
    return oa,aa,kp


def train(train_loader,val_loader,test_loader,**kwargs):
    device=kwargs.get('device')
    lr=kwargs.get('lr')
    epochs=kwargs.get('epochs')
    batch=kwargs.get('batch_size')
    save_name=kwargs.get('save_name')
    cnnNet=myCNN(kwargs.get('input_channels'),kwargs.get('n_classes'),kwargs.get('patch_size'))
    checkpoint=kwargs.get('checkpoint')
    if checkpoint is not None:
        cnnNet.load_state_dict(torch.load(os.path.join(CNN_model_path, checkpoint)))
    cnnNet.to(device)
    criterion = nn.CrossEntropyLoss()
    # todo optimizer 属性修改
    optimizer = optim.SGD(cnnNet.parameters(), lr=lr,momentum=kwargs.get('momentum'))
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, verbose=True,min_lr=1.0e-3)
    print('init paramters:\nepochs:{} lr:{} batch:{} '.format(epochs, lr, batch))
    val_max_oa = 0.0
    for epoch in range(epochs):
        iter=0
        for sub_img,sub_labels,sub_seg in train_loader:
            iter=iter+1
            # sub_img=np.transpose(sub_img,(0,3,1,2))
            sub_seg=np.transpose(sub_seg,(0,3,1,2))

            x=torch.as_tensor(sub_seg.to(device),dtype=torch.float32).to(device)
            target=torch.as_tensor(sub_labels-1,dtype=torch.int64).to(device)
            x = torch.unsqueeze(x, 1)
            # sub_labels = torch.unsqueeze(sub_labels, 0)

            optimizer.zero_grad()
            output=cnnNet(x)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            print('\repoch:{}/{} iter:{}/{} loss:{}'.format(epoch + 1, epochs, iter, train_loader.__len__(), loss), end='')
        # lr_scheduler.step(loss)
        if (epoch+1)%10==0:
            val_oa,val_aa,val_kp= val(val_loader,cnnNet,**kwargs)
            print(' val_OA:{:.6f} val_aa:{:.6f} val_kappa:{:.6f} lr:{:.6f}'.format(val_oa,val_aa,val_kp,optimizer.param_groups[0]['lr']))
            if val_max_oa < val_oa:
                torch.save(cnnNet.state_dict(), os.path.join(CNN_model_path, 'cnn_best_model.pkl'))
                val_max_oa = val_oa
                print('Saving best model...', val_max_oa)

    test_oa,test_aa,test_kp=test(test_loader,cnnNet,**kwargs)
    print('last model: test_oa:{:.6f} test_aa:{:.6f} test_kappa:{:.6f}'.format(test_oa,test_aa,test_kp))
    model_name = 'cnnNet_'+ save_name + '_{}.pkl'.format(test_oa)
    torch.save(cnnNet.state_dict(), os.path.join(CNN_model_path,model_name))

    cnnNet.load_state_dict(torch.load(os.path.join(CNN_model_path, 'cnn_best_model.pkl')))
    test_oa, test_aa, test_kp = test(test_loader, cnnNet, **kwargs)
    print('best model: test_oa:{:.6f} test_aa:{:.6f} test_kappa:{:.6f}'.format(test_oa, test_aa, test_kp))
    torch.save(cnnNet.state_dict(), os.path.join(CNN_model_path, model_name))
    return test_oa,test_aa,test_kp


if __name__ == '__main__':
    kwargs={
        'dataset_name':'IndianPines', # IndianPines,PaviaU
        'save_name': 'IP',

        'patch_size':3,
        'batch_size':1024,
        'checkpoint':'cnnNet_IP_0.9686720867208672.pkl',

        'epochs':1000,
        'lr': 0.1,
        'momentum':0.9
    }

    image, labels, seg_img,kwargs = load_data(kwargs.get('dataset_name'),**kwargs)
    # 取出可用样本索引
    remain_indices = np.argwhere(labels != 0)
    remain_labels=labels[remain_indices[:,0],remain_indices[:,1]]
    # 划分训练集 验证集 测试集
    train_ind,test_ind=train_test_split(remain_indices,stratify=remain_labels,train_size=kwargs.get('train_size'))
    # val_ind,_=train_test_split(test_ind,stratify=labels[test_ind[:,0],test_ind[:,1]],train_size=len(train_ind))
    # 构造数据集
    # train_dataset=mydataset(image,labels,seg_img,train_ind,**kwargs)
    # train_loader=DataLoader(train_dataset,batch_size=kwargs.get('batch_size'),shuffle=True)
    # val_dataset=mydataset(image,labels,seg_img,val_ind,**kwargs)
    # val_loader=DataLoader(val_dataset,batch_size=kwargs.get('batch_size'),shuffle=True)
    test_dataset=mydataset(image,labels,seg_img,test_ind,**kwargs)
    test_loader=DataLoader(test_dataset,batch_size=kwargs.get('batch_size'),shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs.setdefault('device',device)

    cnnNet = myCNN(kwargs.get('input_channels'), kwargs.get('n_classes'), kwargs.get('patch_size'))
    cnnNet.to(device)
    checkpoint=kwargs.get('checkpoint')
    if checkpoint is not None:
        cnnNet.load_state_dict(torch.load(os.path.join(CNN_model_path, checkpoint)))
    oa,aa,kp=test(test_loader,cnnNet,**kwargs)
    print('oa:{:.6f} aa:{:.6f} kappa:{:.6f}'.format(oa,aa,kp))
