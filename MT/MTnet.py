#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 10:08
# @Author  : dongdong
# @File    : main_classify.py
# @Software: PyCharm
# @Purpose :this is for Multi-task HSIC


import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
from sklearn import metrics,preprocessing
import torch.nn as nn
import torch.functional as F
# import torch.optim.optimizer as optimizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import argparse 
nGpus=0
if torch.cuda.device_count()>1:
    nGpus=torch.cuda.device_count()
    print('Use ',nGpus,' GPUs')

root_dir=os.getcwd()  
basedir=os.path.join(root_dir,'MT')# 不同方法的文件夹
# 参数设置 命令行参数，若未指定则为默认值
# python ./MT/MTnet.py --data_name IndianPines --batch_size 1024 --train_size 5 
parser=argparse.ArgumentParser()
parser.add_argument('--device',action='store',type=str,default='cuda:0',required=False,help='cpu|cuda:0')
parser.add_argument('--seed',type=int,default=2021,required=False,help='random seed')
parser.add_argument('--isTest',action='store_true',required=False,help='test or train flag')
# 数据、训练相关参数
parser.add_argument('--Dataset_path',action='store',type=str,default=os.path.join(root_dir,'Datasets'),required=False,help='the datasets path')
parser.add_argument('--data_name',action='store',type=str,default='PaviaU',required=False,help='IndianPines|PaviaU|Salinas|Houston2013')
# -1 表示未指定，将在Args初始化时使用默认值
parser.add_argument('--batch_size',type=int,default='-1',required=False,help='batch_size')
parser.add_argument('--patch_size',type=int,default=-1,required=False,help='patch_size')
parser.add_argument('--train_size',type=float,default=-1.0,required=False,help='train_size')
parser.add_argument('--epochs',type=int,default=-1,required=False,help='epochs')
pargs=parser.parse_args()

Dataset_path=pargs.Dataset_path
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')
seed=pargs.seed
np.random.seed(seed)
torch.manual_seed(seed)
device=pargs.device
if not torch.cuda.is_available():
        device='cpu'

def preprocess(dataset, normalization = 2):
    '''
    对数据集进行归一化；
    normalization = 1 : 0-1归一化；  2：标准化； 3：分层标准化；4.逐点正则化
    '''
    #attation 数据归一化要做在划分训练样本之前；
    dataset = np.array(dataset, dtype = 'float64')
    [m,n,b] = np.shape(dataset)
    #先进行数据标准化再去除背景  效果更好
    if normalization ==1:
        dataset = np.reshape(dataset, [m*n,b])
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)
        dataset = dataset.reshape([m,n,b])
    elif normalization ==2:
        dataset = np.reshape(dataset, [m*n,b,])
        stand_scaler = preprocessing.StandardScaler()
        dataset = stand_scaler.fit_transform(dataset)
        dataset = dataset.reshape([m,n,b])
    elif normalization ==3:
        stand_scaler = preprocessing.StandardScaler()
        for i in range(b):
            dataset[:,:,i] = stand_scaler.fit_transform(dataset[:,:,i])
    elif normalization ==6:
        stand_scaler = preprocessing.MinMaxScaler()
        for i in range(b):
            dataset[:,:,i] = stand_scaler.fit_transform(dataset[:,:,i])
    elif normalization == 4:
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = preprocessing.normalize(dataset[i,j,:].reshape(1,-1))[0]
    elif normalization == 5:
        stand_scaler = preprocessing.StandardScaler()
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = stand_scaler.fit_transform(dataset[i,j,:].reshape(-1,1)).flatten()
    elif normalization == 7:
        stand_scaler = preprocessing.MinMaxScaler()
        for i in range(m):
            for j in range(n):
                dataset[i,j,:] = stand_scaler.fit_transform(dataset[i,j,:].reshape(-1,1)).flatten()
    else:
        pass

    return (dataset)
def my_pca(image,n_components=0.99,**kwargs):
    """对image 的通道 进行pca，并恢复到image"""
    w,h,b=len(image),len(image[0]),len(image[0][0])
    image = np.reshape(image, [w*h, b])
    pca=PCA(n_components=n_components)
    # pca=PCA()
    pca.fit(image)
    return  np.reshape(pca.transform(image),[w,h,-1])
def my_train_test_split(data,labels,train_rate,random_state):
    trainX, testX, trainY, testY=None,None,None,None
    for it in np.unique(labels):
        if it == 0:
            continue
        it_index=np.argwhere(labels==it)[:,0]
        if (train_rate<1.0 and it_index.shape[0]*train_rate<=5):
            itrainX, itestX, itrainY, itestY = train_test_split(data[it_index,:], labels[it_index], train_size=5,
                                                                random_state=random_state)
        elif (train_rate>1.0 and it_index.shape[0]<=train_rate) :
            itrainX, itestX, itrainY, itestY = train_test_split(data[it_index, :], labels[it_index], train_size=10,
                                                                random_state=random_state)
        else:
            itrainX, itestX, itrainY, itestY= train_test_split(data[it_index,:], labels[it_index], train_size=train_rate,
                                                    random_state=random_state)
        trainX=np.concatenate((trainX,itrainX)) if trainX is not None else itrainX
        trainY=np.concatenate((trainY,itrainY)) if trainY is not None else itrainY
        testX=np.concatenate((testX,itestX)) if testX is not None else itestX
        testY=np.concatenate((testY,itestY)) if testY is not None else itestY
    return trainX, testX, trainY, testY

def print_metrics(y_true,y_pred,show=True):
    print('metrics : -------------------------')
    kp= metrics.cohen_kappa_score(y_true,y_pred)
    classify_report = metrics.classification_report(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    acc_for_each_class = metrics.recall_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('-----------------------------------------')
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:',kp)
    return overall_accuracy,average_accuracy,kp,classify_report,confusion_matrix,acc_for_each_class

def saveFig(img,imgname):
    h,w=img.shape
    plt.imshow(img,'jet')
    plt.axis('off')
    plt.gcf().set_size_inches(w / 100.0 / 3.0, h / 100.0 / 3.0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.savefig(imgname,dpi=600)



def load_data(data_name,nC=50):
    
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
        
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        
    elif data_name=='Salinas':
        image = sio.loadmat(os.path.join(Salinas_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(Salinas_path, 'Salinas_gt.mat'))['salinas_gt']


    elif data_name=='Houston2013':
        # seg_name='Houston2013_255seg'+str(nC)+'.mat'
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']

    edge_label_path=os.path.join(Dataset_path,data_name+'_edgelabels.mat')
    edge_dic=sio.loadmat(edge_label_path)
    return np.array(image,dtype=np.float),np.array(labels,dtype=np.int),edge_dic

class MyData():
    def __init__(self,image,labels,edge_dic,train_ind,test_ind,val_ind=[],patch_size=1,batch_size=128):
        w,h,b=image.shape
        
        self.batch_size=batch_size
        self.patch_size=patch_size
        self.channels=image.shape[-1]
        self.image_shape=image.shape
        self.labels=labels
        self.image_pad=np.zeros((w+patch_size-1,h+patch_size-1,b))
        self.image_pad[patch_size//2:w+patch_size//2,patch_size//2:h+patch_size//2,:]=image
        self.edge_dic=edge_dic
        self.edge_dic_keys=['edgerow','edgecol','edgeabove','edgebelow']
        self.xd=[0,1,-1,1]
        self.yd=[1,0,1,1]
        # 分类训练集
        self.train_ind=train_ind
        self.test_ind=test_ind
        self.val_ind=val_ind
        self.train_iters=len(self.train_ind)//batch_size+1
        self.test_iters = len(self.test_ind) // batch_size+1
        self.val_iters=len(self.val_ind)//batch_size+1
        self.val_batch=self.batch_size
        if self.val_iters==1:
            self.val_iters=4
            self.val_batch=len(self.val_ind)//self.val_iters+1 
        self.all_indices=np.concatenate([self.train_ind,self.val_ind,self.test_ind],axis=0)
        self.all_iters = (len(self.all_indices)) // self.batch_size + 1
        
        np.random.shuffle(self.train_ind)
        # 回归训练集(所有的边,横、竖、斜上、斜下)    
        self.regress_ind=self.get_regress_ind()
        np.random.shuffle(self.regress_ind)
        self.reg_iters=len(self.regress_ind)//batch_size+1
        if pargs.data_name=='Houston2013':
            # 对houston2013数据集使用10%的回归数据
            np.random.shuffle(self.regress_ind)
            self.reg_iters=int(self.reg_iters/3)
        
    def get_regress_ind(self):
        # 回归训练集(所有的边,横、竖、斜上、斜下)    
        # edgerow=np.zeros((w,h)) # row,col range:[0,w-1],[0,h-2] 可取到 cnt=(w)*(h-1)
        # edgecol=np.zeros((w,h))# row,col range:[0,w-2],[0,h-1]  cnt=(w-1)*(h)
        # edgeabove=np.zeros((w,h))# row,col range:[1,w-1],[0,h-2] cnt=(w-1)*(h-1)
        # edgebelow=np.zeros((w,h))# row,col range:[0,w-2],[0,h-2] cnt=(w-1)*(h-1)
        # 用[a,x,y]表示位置[x,y] Type A的边 
        w,h=self.image_shape[0:2]
        grids=[]
        a,b,c=np.mgrid[0:1,0:w,0:h-1]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        a,b,c=np.mgrid[1:2,0:w-1,0:h]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        a,b,c=np.mgrid[2:3,1:w,0:h-1]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        a,b,c=np.mgrid[3:4,0:w-1,0:h-1]
        grid=np.c_[a.ravel(),b.ravel(),c.ravel()]
        grids.append(grid)
        grids=np.concatenate([grids[0],grids[1],grids[2],grids[3]],axis=0)

        # edges_ind=[]
        
        # for i in range(w):
        #     for j in range(h):
        #         if j!=h-1:# edge row
        #             edges_ind.append([0,i,j])
        #         if i!=w-1:# edge col 
        #             edges_ind.append([1,i,j])
        #         if i!=0 and j!=h-1:# edge above
        #             edges_ind.append([2,i,j])
        #         if i!=w-1 and j!=h-1:# edge below
        #             edges_ind.append([3,i,j])
        return grids 

    def get_batch_reg(self,iter):
        if iter == self.reg_iters-1:
            indices=self.regress_ind[-self.batch_size:]
        else:
            indices=self.regress_ind[iter*self.batch_size:(iter+1)*self.batch_size]
        hp=self.patch_size
        X=np.zeros((self.batch_size,hp,hp,self.channels),dtype=np.float32)
        X2=np.zeros((self.batch_size,hp,hp,self.channels),dtype=np.float32)
        Yc=np.zeros((self.batch_size,),dtype=np.float32)
        Yc2=np.zeros((self.batch_size,),dtype=np.float32)
        Yr=np.zeros((self.batch_size,1),dtype=np.float32)
        for ind_i,edge_i in enumerate(indices):
            edge_type,edge_pos=self.edge_dic_keys[edge_i[0]],edge_i[1:]
            Yr[ind_i]=self.edge_dic[edge_type][edge_pos[0],edge_pos[1]]

            a,b=edge_pos
            X[ind_i,:,:,:]=self.image_pad[a:a+hp,b:b+hp,:]
            if [a,b] in self.train_ind:
                Yc[ind_i]=self.labels[a,b]
            else:
                Yc[ind_i]=-1
            a=edge_pos[0]+self.xd[edge_i[0]]
            b=edge_pos[1]+self.yd[edge_i[0]] 
            X2[ind_i,:,:,:]=self.image_pad[a:a+hp,b:b+hp,:]
            if [a,b] in self.train_ind:
                Yc2[ind_i]=self.labels[a,b]
            else:
                Yc2[ind_i]=-1
        return np.array(X,dtype=np.float32),np.array(Yc,dtype=np.float)-1,\
            np.array(X2,dtype=np.float32),np.array(Yc2,dtype=np.float)-1,\
            np.array(Yr,dtype=np.float)
    
    def get_batch(self,iter,testOrval='test'):
        '''testOrval:test|val|all'''
        tmp_batch=self.batch_size
        if testOrval=='test':
            if iter == self.test_iters - 1:
                # indices = self.test_ind[-self.batch_size:]
                indices = self.test_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.test_ind[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif testOrval=='val':
            tmp_batch = self.val_batch
            if iter == self.val_iters - 1:
                indices = self.val_ind[-self.val_batch:]
                # indices = self.val_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.val_ind[iter * self.val_batch:(iter + 1) * self.val_batch]
        elif testOrval=='all':
            if iter == self.all_iters - 1:
                # indices = self.all_indices[-self.batch_size:]
                indices = self.all_indices[iter * self.batch_size:]
                tmp_batch=len(indices)
            else:
                indices = self.all_indices[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif testOrval=='train':
            # the val_batch also serves as the tmp_batch of trainset, as train and val are same small size 
            tmp_batch = self.val_batch
            if iter == self.train_iters - 1:
                indices = self.train_ind[-self.val_batch:]
                # indices = self.val_ind[iter * self.batch_size:]
                tmp_batch = len(indices)
            else:
                indices = self.train_ind[iter * self.val_batch:(iter + 1) * self.val_batch]

        X = np.zeros((tmp_batch, self.patch_size, self.patch_size, self.channels))
        Y = np.zeros((tmp_batch,))
        hp = self.patch_size

        for ind_i, ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i, :, :, :] = self.image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            Y[ind_i] = self.labels[ind[0], ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1,indices


    def shuffle(self):
        np.random.shuffle(self.train_ind)
        np.random.shuffle(self.regress_ind)

'''
    def get_batch_twoimage(self,iter,fortype='train'):
        # return the patches of 'iter' batch in page_i of image

        if fortype=='train':
            if iter==self.train_iters-1:
                indices = self.train_ind[-self.batch_size:]
            else:
                indices=self.train_ind[iter*self.batch_size:(iter+1)*self.batch_size]
        elif fortype=='test':
            if iter == self.test_iters-1:
                indices = self.test_ind[-self.batch_size:]
            else:
                indices = self.test_ind[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif fortype=='val':
            if iter == self.val_iters-1:
                indices = self.val_ind[-self.batch_size:]
            else:
                indices = self.val_ind[iter * self.batch_size:(iter + 1) * self.batch_size]

        X=np.zeros((self.batch_size,self.patch_size,self.patch_size,self.channels))
        XS=np.zeros((self.batch_size,self.patch_size,self.patch_size,self.channels))
        Y=np.zeros((self.batch_size,))
        hp=self.patch_size

        for ind_i,ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i,:,:,:]=self.image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            XS[ind_i,:,:,:]=self.seg_image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            Y[ind_i]=self.labels[ind[0],ind[1]]
        return np.array(X,dtype=np.float32),np.array(XS,dtype=np.float32),np.array(Y,dtype=np.float)-1

    def get_batch_oneimage(self,iter,imagename='image',fortype='train'):
        # return the patches of 'iter' batch of image

        if fortype=='train':
            if iter==self.train_iters-1:
                indices = self.train_ind[-self.batch_size:]
            else:
                indices=self.train_ind[iter*self.batch_size:(iter+1)*self.batch_size]
        elif fortype=='test':
            if iter == self.test_iters-1:
                indices = self.test_ind[-self.batch_size:]
            else:
                indices = self.test_ind[iter * self.batch_size:(iter + 1) * self.batch_size]
        elif fortype=='val':
            if iter == self.val_iters-1:
                indices = self.val_ind[-self.batch_size:]
            else:
                indices = self.val_ind[iter * self.batch_size:(iter + 1) * self.batch_size]

        X=np.zeros((self.batch_size,self.patch_size,self.patch_size,self.channels))
        # XS=np.zeros((self.batch_size,self.patch_size,self.patch_size,self.channels))
        Y=np.zeros((self.batch_size,))
        hp=self.patch_size

        for ind_i,ind in enumerate(indices):
            # Pay attention to coordinate changes
            if imagename == 'image':
                X[ind_i,:,:,:]=self.image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            else:
                X[ind_i,:,:,:]=self.seg_image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            Y[ind_i]=self.labels[ind[0],ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1,indices

    def get_all(self,iter):
        if iter == self.all_iters - 1:
            indices = self.all_indices[-self.batch_size:]
        else:
            indices = self.all_indices[iter * self.batch_size:(iter + 1) * self.batch_size]

        X = np.zeros((self.batch_size, self.patch_size, self.patch_size, self.channels))
        XS = np.zeros((self.batch_size, self.patch_size, self.patch_size, self.channels))
        Y = np.zeros((self.batch_size))
        hp = self.patch_size

        for ind_i, ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i, :, :, :] = self.image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            XS[ind_i, :, :, :] = self.seg_image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            Y[ind_i] = self.labels[ind[0], ind[1]]
        return np.array(X,dtype=np.float32),np.array(XS,dtype=np.float32),np.array(Y,dtype=np.float)-1,indices

'''


class MyNet(nn.Module):
    def __init__(self,out_dim,bands,patch_size=1,batch_size=128):
        super(MyNet,self).__init__()
        self.patch_size=patch_size
        self.batch_size=batch_size
        self.bands=bands
        self.out_dim=out_dim
        self.batch_nGpus=self.batch_size//nGpus if nGpus>1 else self.batch_size
        self.conv3dseq=nn.Sequential(
            nn.Conv3d(1,8,kernel_size=(7,3,3),padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(5,3, 3),padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3),padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        
        self.f_shape1=self.get_fea_nums()# (batch,channel,w,h)
        self.conv2d1=nn.Sequential(
            nn.Conv2d(self.f_shape1[1],64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        fc_in=64*self.f_shape1[2]*self.f_shape1[3]
        self.fcseq=nn.Sequential(
            nn.Linear(fc_in,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,self.out_dim)
        )
        self.fcreg=nn.Sequential(
            nn.Linear(fc_in,256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,1)
        )

    def forward(self,X1,X2,isTest=False):
        if isTest:
            X1=self.conv3dseq(X1)
            X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
            X1=self.conv2d1(X1)
            X1 = X1.view(X1.shape[0], -1)
            X1=self.fcseq(X1)
            return X1

        X1=self.conv3dseq(X1)
        X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X1=self.conv2d1(X1)
        X1 = X1.view(self.batch_nGpus, -1)

        X2=self.conv3dseq(X2)
        X2=X2.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X2=self.conv2d1(X2)
        X2 = X2.view(self.batch_nGpus, -1)

        Xr=X1+X2
        Xr=self.fcreg(Xr)
        X1=self.fcseq(X1)
        X2=self.fcseq(X2)
        
        return X1,X2,Xr

    def get_fea_nums(self):
        x=torch.ones(self.batch_size,1,self.bands,self.patch_size,self.patch_size)
        x=self.conv3dseq(x)
        _,_,_,win_w,win_h=x.shape
        a=x.view([self.batch_size,-1,win_w,win_h]).shape
        return a
    
    def test_forward(self,X1):
        X1=self.conv3dseq(X1)
        X1=X1.view(-1,self.f_shape1[1],self.f_shape1[2],self.f_shape1[3])
        X1=self.conv2d1(X1)
        X1 = X1.view(self.batch_nGpus, -1)
        X1=self.fcseq(X1)
        return X1


def test_and_show(args,mydata=None,colormap=None):
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # args = Args()
    # load data
    if mydata is None:
        # load data
        # load data
        image_raw, labels_raw, edge_dic= load_data(args.data_name)
        remain_index = np.where(labels_raw != 0)
        remain_index = np.array(list(zip(remain_index[0], remain_index[1])))
        train_ind, test_ind, _, _ = my_train_test_split(remain_index, labels_raw[remain_index[:, 0], remain_index[:, 1]],
                                                        args.train_size, random_state=seed)
        val_ind, test_ind = train_test_split(test_ind, train_size=len(train_ind), random_state=seed)
        # val_ind=train_ind
        print('train samples:', len(train_ind))
        a, b = np.unique(labels_raw[train_ind[:, 0], train_ind[:, 1]], return_counts=True)
        print(list(zip(a, b)))
        print('test samples:', len(test_ind))
        a, b = np.unique(labels_raw[test_ind[:, 0], test_ind[:, 1]], return_counts=True)
        print(list(zip(a, b)))


        # image_pca=my_pca(image_raw,30)
        # seg_img_pca=my_pca(seg_img,30)

        # maybe pca,so we set name with xxx_pca
        image_pca = image_raw
        image_pca = preprocess(image_pca)

        mydata = MyData(image_pca, labels_raw,edge_dic, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)
        # prepare Net
        bands_num = image_pca.shape[-1]
        myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
        if torch.cuda.device_count()>1:
            print('Use ',torch.cuda.device_count() ,' GPUs')
            myNet=nn.DataParallel(myNet)
        myNet.to(device)
        
        modelname = args.method_name + args.data_name + '_lossbst' + '.model'
        myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
            # todo one or two image :net

    print('test all...')
    myNet.eval()

    y_pred = []
    y_true = []
    indices=[]
    iters = mydata.all_iters
    for iter in tqdm(range(iters),desc='test all:'):
        X1, Yc1,ind= mydata.get_batch(iter,'all')

        X1 = np.transpose(X1, [0, 3, 1, 2])
        X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
        X1 = X1.unsqueeze(1)
        output = myNet(X1,X1,isTest=True)
        # 处理无标签分类结果
        

        y_pred.extend(torch.argmax(output.detach(), 1).cpu().numpy().tolist())
        y_true.extend(Yc1.tolist())
        indices.extend(ind.tolist())

    indices=np.array(indices)
    labels_show=np.zeros(mydata.image_shape[0:2])
    labels_show[indices[:,0],indices[:,1]]=np.array(y_pred)+1

    all_oa=metrics.accuracy_score(y_true,y_pred)
    print(all_oa)
    writelogtxt(args.logFile,'OA:'+str(all_oa))
    print_metrics(y_true,y_pred)
    figname = modelname.split('.')[0] + "_single.png"
    saveFig(labels_show,figname)
    figname = modelname.split('.')[0] + "_gt.png"
    saveFig(labels_raw,figname)
    # plt.imshow(labels_show, cmap='jet')
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.savefig(figname,dpi=600)

    # plt.subplot(121)
    # plt.axis('off')
    # plt.margins(0,0)
    # # plt.imshow(mydata.labels, cmap='tab20')
    # plt.imshow(mydata.labels, cmap='jet')
    # plt.title('true labels')
    # plt.subplot(122)
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.imshow(labels_show, cmap='jet')
    # plt.title('predict labels')
    # # plt.imshow(labels_show, cmap='tab20')
    # figname=modelname.split('.')[0]+".png"
    # plt.savefig(figname,dpi=600)
    # # plt.show()
    pass

def test_val(mydata:MyData,myNet:MyNet,fortype='test'):
    print(fortype,'...')
    myNet.eval()
    # device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    y_pred=[]
    y_true=[]
    if fortype=='test':
        iters=mydata.test_iters
    elif fortype=='val':
        iters=mydata.val_iters
    else:
        print("error test_val args:fortype.")
        exit()

    for iter in range(iters):
        X1, Yc1,ind= mydata.get_batch(iter,fortype)
        X1 = np.transpose(X1, [0, 3, 1, 2])
        X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
        X1 = X1.unsqueeze(1)
        output = myNet(X1,X1,isTest=True)# 使用val_batch 避免batch大于验证样本数量时的错误

        y_pred.extend(torch.argmax(output.detach(),1).cpu().numpy().tolist())
        y_true.extend(Yc1.tolist())
    acc=metrics.accuracy_score(y_true,y_pred)
    return acc,y_pred,y_true


def run(args=None):

    print("run with args dict:", args.__dict__)
    # load data
    image_raw, labels_raw, edge_dic= load_data(args.data_name)
    remain_index = np.where(labels_raw != 0)
    remain_index = np.array(list(zip(remain_index[0], remain_index[1])))
    train_ind, test_ind, _, _ = my_train_test_split(remain_index, labels_raw[remain_index[:, 0], remain_index[:, 1]],
                                                    args.train_size, random_state=seed)
    val_ind, test_ind = train_test_split(test_ind, train_size=len(train_ind), random_state=seed)
    # val_ind=train_ind
    print('train samples:', len(train_ind))
    a, b = np.unique(labels_raw[train_ind[:, 0], train_ind[:, 1]], return_counts=True)
    print(list(zip(a, b)))
    print('test samples:', len(test_ind))
    a, b = np.unique(labels_raw[test_ind[:, 0], test_ind[:, 1]], return_counts=True)
    print(list(zip(a, b)))
    

    # image_pca=my_pca(image_raw,30)
    # seg_img_pca=my_pca(seg_img,30)

    # maybe pca,so we set name with xxx_pca
    image_pca = image_raw
    image_pca = preprocess(image_pca)

    mydata = MyData(image_pca, labels_raw,edge_dic, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)
    # prepare Net
    bands_num = image_pca.shape[-1]
    myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
    if torch.cuda.device_count()>1:
        print('Use ',torch.cuda.device_count() ,' GPUs')
        myNet=nn.DataParallel(myNet)
    myNet.to(device)
    if args.checkpoint==True:
        modelname = args.method_name + args.data_name + '_lossbst' + '.model'
        if os.path.exists(os.path.join(args.model_save_path, modelname)):
            myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
            print('load model from ',modelname)
            writelogtxt(args.logFile,'load model from '+modelname)
        
    # modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    # myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    # test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    # print('test_acc:', test_acc)
    # print_metrics(y_true,y_pred)

    CE = nn.CrossEntropyLoss()
    MSE=nn.MSELoss()
    optim = torch.optim.SGD(myNet.parameters(), lr=args.lr, momentum=0.9)
    # optim = torch.optim.Adagrad(myNet.parameters(), lr=0.0005, weight_decay=0.01)

    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in myNet.state_dict():
        print(param_tensor, "\t", myNet.state_dict()[param_tensor].size())
    val_bst = 0
    loss_bst=-1
    bst_epoch = 0
    for epoch in range(args.epochs):
        myNet.train()
        mydata.shuffle()
        for iter in range(mydata.reg_iters):
            optim.zero_grad()
            X1, Yc1, X2,Yc2,Yr = mydata.get_batch_reg(iter)

            X1 = np.transpose(X1, [0, 3, 1, 2])
            X2 = np.transpose(X2, [0, 3, 1, 2])
            X1 = torch.from_numpy(X1).requires_grad_(True).to(device)
            X2 = torch.from_numpy(X2).requires_grad_(True).to(device)
            X1 = X1.unsqueeze(1)
            X2 = X2.unsqueeze(1)
            outYc1,outYc2,outYr = myNet(X1, X2)
            # 处理无标签分类结果,两种选择：*1认为模型分的都对，
            tmp=np.where(Yc1<0)
            Yc1[tmp]=np.argmax(outYc1.detach().cpu().numpy(),axis=1)[tmp]
            tmp=np.where(Yc2<0)
            Yc2[tmp]=np.argmax(outYc2.detach().cpu().numpy(),axis=1)[tmp]

            Yc1= torch.from_numpy(Yc1).long().to(device)
            Yc2= torch.from_numpy(Yc2).long().to(device)
            Yr= torch.from_numpy(Yr).float().to(device)

            loss1=CE(outYc1, Yc1) 
            loss2=CE(outYc2, Yc2) 
            lossr=MSE(outYr, Yr)
            loss = loss1+loss2+lossr 
            loss.backward()
            optim.step()
            logtext='\repoch:{}/{} iter:{}/{} loss:{:.6f} lossr:{:.6f} loss1:{:.6f} loss2:{:.6f}'\
                .format(epoch, args.epochs, iter, mydata.reg_iters,loss.detach().cpu(), \
                    lossr.detach().cpu(),loss1.detach().cpu(),loss2.detach().cpu())
            print(logtext,end='')
            # writelogtxt(args.logFile,logtext[1:])
        logtext='\repoch:{}/{} iter:{}/{} loss:{:.6f} lossr:{:.6f} loss1:{:.6f} loss2:{:.6f}'\
            .format(epoch, args.epochs, iter, mydata.reg_iters,loss.detach().cpu(), \
                lossr.detach().cpu(),loss1.detach().cpu(),loss2.detach().cpu())
        print(logtext,end='')
        writelogtxt(args.logFile,logtext[1:])

        if (epoch + 1) % 1 == 0:
            # save best model in val_set
            # val_acc, _, _ = test_val(mydata, myNet, 'val')
            # print('val_acc:', val_acc)
            # losscpu=lossr.detach().cpu()
            losscpu=loss1.detach().cpu()/2+loss2.detach().cpu()/2
            loss_bst=losscpu if loss_bst==-1 else loss_bst
            if losscpu <= loss_bst:
                loss_bst = losscpu
                bst_epoch = epoch + 1
                modelname = args.method_name + args.data_name + '_lossbst' + '.model'
                torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
                print("save model:", modelname)
                writelogtxt(args.logFile,"save model:"+ modelname)
            if epoch - bst_epoch > args.early_stop_epoch:
                print('train break since val_acc does not improve from epoch: ', bst_epoch)
                writelogtxt(args.logFile,'train break since val_acc does not improve from epoch: '+str(bst_epoch))
                break
        # if (epoch + 1) % 5 == 0:
            test_acc, y_pred, y_true = test_val(mydata, myNet, 'val')
            print('val_acc:', test_acc)
            writelogtxt(args.logFile,'val_acc:'+str(test_acc))

    # test on best model
    modelname = args.method_name+ args.data_name + '_lossbst' + '.model'
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    print('test_acc:', test_acc)
    writelogtxt(args.logFile,'test_acc:'+str(test_acc))

    overall_accuracy, average_accuracy, kp, classify_report, confusion_matrix, acc_for_each_class = print_metrics(
        y_true, y_pred)

    a, b = np.unique(y_true, return_counts=True)
    print('y_true cnt:', list(zip(a, b)))
    a, b = np.unique(y_pred, return_counts=True)
    print('y_pred cnt:', list(zip(a, b)))
    return overall_accuracy, average_accuracy, kp, acc_for_each_class


def writelogtxt(logfilepath,content,with_time=True):
    f = open(logfilepath, 'a')
    # f.write(time.asctime())
    tm_str=""
    if with_time:
        tm_str = str(time.asctime())+" "
    f.write(tm_str+content+'\n')
    f.flush()
    f.close()


class Args():
    def __init__(self, data_name='Salinas'):
        self.method_name = 'MT9_'
        self.data_name = pargs.data_name
        self.batch_size = 1024
        self.epochs = 200
        self.lr=0.0005
        # 在set_dataset()中，相关参数会更新为命令行指定参数，所以放在最后
        self.set_dataset(self.data_name)
        self.model_save_path = os.path.join(basedir,'models')

        self.early_stop_epoch = 30
        self.checkpoint=True
        self.record = True
        self.recordFile = self.method_name+'_'+self.data_name+'_results.txt'
        self.logfilepath=os.path.join(basedir,'logs')
        self.logFile=os.path.join(self.logfilepath,self.method_name+self.data_name+'_logs.txt')

    def set_dataset(self, data_name):
        if data_name == 'PaviaU':
            self.data_name = 'PaviaU'
            self.train_size = 5
            self.classnums = 9
            self.patch_size = 5
        elif data_name == 'IndianPines':
            self.data_name = 'IndianPines'
            self.train_size = 5
            self.classnums = 16
            self.patch_size = 5
        elif data_name == 'Salinas':
            self.data_name = 'Salinas'
            self.train_size = 5
            self.classnums = 16
            self.patch_size = 5
        elif data_name == 'Houston2013':
            self.data_name = 'Houston2013'
            self.train_size = 5
            self.classnums = 15
            self.patch_size = 5
        # 如果参数在命令行中指定，则将参数更新为命令行指定参数
        if pargs.train_size != -1:
            self.train_size=pargs.train_size
            if self.train_size>=1:
                self.train_size=int(self.train_size)
        if pargs.patch_size != -1:
            self.patch_size=pargs.patch_size
        if pargs.epochs != -1:
            self.epochs=pargs.epochs
        if pargs.batch_size != -1:
            self.batch_size=pargs.batch_size

def run_more(args:Args):
    
    writelogtxt(args.logFile,"run more start!")
    print("run more start! ")

    global  seed
    oas = []
    aas = []
    kps = []
    afes = []
    for run_id in range(5):
        
        # seed=run_id+2000
        seed = np.random.randint(10000)
        args.checkpoint=False
        print(" reset the checkpoint to False! ")
        oa, aa, kp, acc_for_each_class = run(args)
        print('run_id:', run_id)
        print('oa:{} aa:{} kp:{}'.format(oa, aa, kp))
        writelogtxt(args.logFile,"run id:"+str(run_id))
        writelogtxt(args.logFile,'oa:{} aa:{} kp:{}'.format(oa, aa, kp))
        oas.append(oa)
        aas.append(aa)
        kps.append(kp)
        afes.append(acc_for_each_class)
    print('afes', afes)
    print('oas:', oas)
    print('aas:', aas)
    print('kps', kps)
    writelogtxt(args.logFile,"afes: "+str(afes))
    writelogtxt(args.logFile,"oas: "+str(oas))
    writelogtxt(args.logFile,"aas: "+str(aas))
    writelogtxt(args.logFile,"kps: "+str(kps))
    print('mean and std:oa,aa,kp')
    print(np.mean(oas) * 100, np.mean(aas) * 100, np.mean(kps) * 100)
    print(np.std(oas), np.std(aas), np.std(kps))
    print('mean/std oa for each class,axis=0:')
    writelogtxt(args.logFile,"mean:oa,aa,kp: "+str(np.mean(oas) * 100)+" "+str(np.mean(aas) * 100)+" "
        + str(np.mean(kps) * 100))
    writelogtxt(args.logFile,"std:oa,aa,kp: "+str(np.std(oas) * 100)+" "+str(np.std(aas) * 100)+" "
        + str(np.std(kps) * 100))
    m_afes = np.mean(afes, axis=0)
    for a in m_afes:
        print(a * 100)
        writelogtxt(args.logFile,str(a * 100),with_time=False)
    print(np.mean(oas) * 100)
    print(np.mean(aas) * 100)
    print(np.mean(kps) * 100)
    print(np.std(oas) * 100)
    print(np.std(aas) * 100)
    print(np.std(kps) * 100)
    print(np.std(afes, axis=0))
    writelogtxt(args.logFile,"run more over !!")
    if args.record:
        f = open(args.recordFile, 'a')
        f.write('\n***********************************\n')
        f.write(time.asctime())
        f.write('\n-------------------\n')
        f.write(str(args.__dict__))
        f.write('\n-------------------\n')
        f.flush()
        f.write('mean/std oa for each class,axis=0:\n')
        for a in m_afes:
            f.write(str(a * 100) + '\n')
        f.flush()
        nplist = [np.mean(oas) * 100, np.mean(aas) * 100, np.mean(kps) * 100,
                  np.std(oas) * 100, np.std(aas) * 100, np.std(kps) * 100]
        for a in nplist:
            f.write(str(a) + '\n')
        f.flush()
        f.close()


if __name__ == '__main__':
    
    args=Args()
    args.set_dataset(pargs.data_name)
    # args.set_dataset("PaviaU")
    if pargs.isTest:
        writelogtxt(args.logFile," a new test all \n"+'*'*50)
    else:
        writelogtxt(args.logFile, " a new start \n" + '*' * 50)
    writelogtxt(args.logFile,str(args.__dict__))
    if not pargs.isTest:
        # run(args)
        run_more(args)
    
    

    test_and_show(args)
    exit()

    # patch_grid=[3,5,7,9,11]
    # nC_grid=[30,50,70,100,200]
    # for patch in patch_grid:
    #     for nC in nC_grid:
    #         args = Args()
    #         # args.set_dataset('IndianPines')
    #         args.set_dataset('PaviaU')
    #         args.patch_size = patch
    #         args.base_nC=nC
    #         args.method_name=args.method_name+'p{}_nC{}'.format(args.patch_size,args.base_nC)
    #         print('running: patch:{} nC:{}'.format(args.patch_size,args.base_nC))
    #         run_more(args)




