#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/3/2 10:08
# @Author  : dongdong
# @File    : main_classify.py
# @Software: PyCharm
# @Purpose :this is for two-image-Net


import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
from sklearn import metrics,preprocessing
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.functional as F
# import torch.optim.optimizer as optimizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
Dataset_path='.\\Datasets'
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')

seed=2021
np.random.seed(seed)
torch.manual_seed(seed)
device='cpu'
if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device='cuda:0'

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

def load_data(data_name,nC=50,show=False):
    # seg_name = data_name + '_seg' + '.mat'
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
        # Seg_img=sio.loadmat(os.path.join(IndianPine_path,'Indian_seg.mat'))['seg_labels']
        # Seg_img=sio.loadmat(os.path.join(IndianPine_path,seg_name))['results']
        # train_rate=0.05
        channel_show=10
        # seg_pca1=sio.loadmat(os.path.join(IndianPine_path,'IndianPines_pca50.mat'))['labels']
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        # Seg_img=sio.loadmat(os.path.join(PaviaU_path,seg_name))['results']
        # train_rate=0.03
        channel_show=100
        # seg_pca1=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_pca50.mat'))['labels']
    elif data_name=='Salinas':
        image = sio.loadmat(os.path.join(Salinas_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(Salinas_path, 'Salinas_gt.mat'))['salinas_gt']
        # Seg_img = sio.loadmat(os.path.join(Salinas_path, seg_name))['results']
        channel_show = 10

    elif data_name=='Houston2013':
        # seg_name='Houston2013_255seg'+str(nC)+'.mat'
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']
        # Seg_img = sio.loadmat(os.path.join(Houston2013_path, seg_name))['results']
        channel_show = 10

    # Seg_img=Seg_img+1
    if show:
        plt.subplot(121)
        plt.title('raw data '+str(channel_show)+'th channel')
        plt.imshow(image[:,:,channel_show])
        plt.subplot(122)
        plt.title('true labels')
        plt.imshow(labels)
        # plt.subplot(123)
        # plt.title('Seg data '+str(channel_show)+'th channel')
        # plt.imshow(Seg_img[:,:,channel_show])
        plt.show()
    return np.array(image,dtype=np.float),np.array(labels,dtype=np.int)
class MyData():
    def __init__(self,image,labels,train_ind,test_ind,val_ind=[],patch_size=1,batch_size=128):
        w,h,b=image.shape
        # ws,hs,bs=seg_image.shape
        self.batch_size=batch_size
        self.patch_size=patch_size
        self.channels=image.shape[-1]
        self.image_shape=image.shape
        # self.seg_image_shape=seg_image.shape
        self.train_ind=train_ind
        self.test_ind=test_ind
        self.val_ind=val_ind
        self.labels=labels
        self.image_pad=np.zeros((w+patch_size-1,h+patch_size-1,b))
        self.image_pad[patch_size//2:w+patch_size//2,patch_size//2:h+patch_size//2,:]=image
        # self.seg_image_pad=np.zeros((w+patch_size-1,h+patch_size-1,bs))
        # self.seg_image_pad[patch_size//2:w+patch_size//2,patch_size//2:h+patch_size//2,:]=seg_image

        np.random.shuffle(self.train_ind)
        self.train_iters=len(self.train_ind)//batch_size+1
        self.test_iters = len(self.test_ind) // batch_size+1
        self.val_iters=len(self.val_ind)//batch_size+1
        self.all_iters=(len(self.train_ind)+len(self.test_ind)+len(self.val_ind))//self.batch_size+1
        self.all_indices=np.concatenate([self.train_ind,self.val_ind,self.test_ind],axis=0)


    def get_batch_twoimage(self,iter,fortype='train'):
        '''return the patches of 'iter' batch in page_i of image'''

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
        '''return the patches of 'iter' batch of image'''

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
                pass
                # X[ind_i,:,:,:]=self.seg_image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            Y[ind_i]=self.labels[ind[0],ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1,indices

    def get_all(self,iter):
        if iter == self.all_iters - 1:
            indices = self.all_indices[-self.batch_size:]
        else:
            indices = self.all_indices[iter * self.batch_size:(iter + 1) * self.batch_size]

        X = np.zeros((self.batch_size, self.patch_size, self.patch_size, self.channels))
        # XS = np.zeros((self.batch_size, self.patch_size, self.patch_size, self.channels))
        Y = np.zeros((self.batch_size,))
        hp = self.patch_size

        for ind_i, ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i, :, :, :] = self.image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            # XS[ind_i, :, :, :] = self.seg_image_pad[ind[0]:ind[0] + hp, ind[1]:ind[1] + hp, :]
            Y[ind_i] = self.labels[ind[0], ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1,indices

    def shuffle(self):
        np.random.shuffle(self.train_ind)

class MyNet(nn.Module):
    def __init__(self,out_dim,bands,patch_size=1,batch_size=128):
        super(MyNet,self).__init__()
        self.patch_size=patch_size
        self.batch_size=batch_size
        self.bands=bands
        # self.seg_bands=seg_bands
        self.out_dim=out_dim
        self.convSeq1=nn.Sequential(
            nn.Conv3d(1,24,(7,1,1),(2,1,1)),
            nn.BatchNorm3d(24),
            nn.ReLU()
        )
        self.convblock1=nn.Sequential(
            nn.Sequential(
                nn.Conv3d(24, 24, (7, 1, 1),(1,1,1),padding=(3,0,0)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(24, 24, (7, 1, 1),(1,1,1),padding=(3,0,0)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        )
        self.convblock2=nn.Sequential(
            nn.Sequential(
                nn.Conv3d(24, 24, (7, 1, 1),(1,1,1),padding=(3,0,0)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(24, 24, (7, 1, 1),(1,1,1),padding=(3,0,0)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        )
        self.num1=self.get_fea_nums()[2]
        self.spc_conv=nn.Sequential(
                nn.Conv3d(24, 128, (self.num1, 1, 1),(1,1,1)),
                nn.BatchNorm3d(128),
                nn.ReLU()
            )
        self.convSeq2= nn.Sequential(
                nn.Conv3d(1, 24, (128, 3, 3),(1,1,1)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        self.convblock31=nn.Sequential(
                nn.Conv3d(1, 24, (24, 3, 3),(1,1,1),padding=(0,1,1)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        self.convblock32=nn.Sequential(
                nn.Conv3d(1, 24, (24, 3, 3),(1,1,1),padding=(0,1,1)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )

        self.convblock41=nn.Sequential(
                nn.Conv3d(1, 24, (24, 3, 3),(1,1,1),padding=(0,1,1)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        self.convblock42= nn.Sequential(
                nn.Conv3d(1, 24, (24, 1, 1),(1,1,1),padding=(0,1,1)),
                nn.BatchNorm3d(24),
                nn.ReLU()
            )
        self.poolsize=self.get_pool_nums()[-1]
        self.pool=nn.AvgPool3d((1,self.poolsize,self.poolsize))
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(24,self.out_dim)
        )

    def forward(self,X):
        X=self.convSeq1(X)
        X=X+self.convblock1(X)

        X=X+self.convblock2(X)
        X=self.conv_squeeze(self.spc_conv,X)
        X=self.conv_squeeze(self.convSeq2,X)
        # X=self.spc_conv(X)
        # X=torch.squeeze(X)
        # X=torch.unsqueeze(X,dim=1)
        X = self.conv_squeeze(self.convblock31, X)
        X = self.conv_squeeze(self.convblock32, X)
        X = self.conv_squeeze(self.convblock41, X)
        X=self.convblock42(X)
        X = torch.squeeze(X)
        X=self.pool(X)
        X = torch.squeeze(X)
        X=self.fc(X)
        return X
    def conv_squeeze(self,conv,X):
        X=conv(X)
        X=torch.squeeze(X)
        X=torch.unsqueeze(X,dim=1)
        return X

    def get_pool_nums(self):
        X=torch.ones(self.batch_size,1,self.bands,self.patch_size,self.patch_size)
        X=self.convSeq1(X)
        X=X+self.convblock1(X)
        X=X+self.convblock2(X)
        X=self.conv_squeeze(self.spc_conv,X)
        X=self.conv_squeeze(self.convSeq2,X)
        X = self.conv_squeeze(self.convblock31, X)
        X = self.conv_squeeze(self.convblock32, X)
        X = self.conv_squeeze(self.convblock41, X)
        X=self.convblock42(X)
        X = torch.squeeze(X)
        return X.shape
    def get_fea_nums(self):
        X=torch.ones(self.batch_size,1,self.bands,self.patch_size,self.patch_size)
        X = self.convSeq1(X)
        X = X + self.convblock1(X)
        X = X + self.convblock2(X)

        return X.shape

def test_and_show(args,mydata=None,colormap=None):
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # args = Args()
    # load data
    if mydata is None:
        image_raw, labels_raw = load_data(args.data_name)
        remain_index = np.where(labels_raw != 0)
        remain_index = np.array(list(zip(remain_index[0], remain_index[1])))
        train_ind, test_ind, _, _ = my_train_test_split(remain_index,
                                                        labels_raw[remain_index[:, 0], remain_index[:, 1]],
                                                        args.train_size, random_state=seed)
        val_ind, test_ind = train_test_split(test_ind, train_size=len(train_ind), random_state=seed)
        # test_ind=remain_index
        image_pca = image_raw
        # seg_img_pca = seg_img
        image_pca = preprocess(image_pca)
        # seg_img_pca = preprocess(seg_img_pca, 1)  # 2 0.9138
        mydata = MyData(image_pca, labels_raw, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)


    bands_num = mydata.image_shape[-1]
    # bands_seg_num = mydata.seg_image_shape[-1]
    myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
    myNet.to(device)

    modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    print('load model:',modelname)
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))

    print('test all...')
    myNet.eval()
    modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    # test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    # print('test_acc:', test_acc)

    # print("0000")
    y_pred = []
    y_true = []
    indices=[]
    iters = mydata.all_iters
    for iter in tqdm(range(iters),desc='test all:'):
        X, Y ,ind= mydata.get_all(iter)
        X = np.transpose(X, [0, 3, 1, 2])
        X = torch.from_numpy(X).requires_grad_(True).to(device)
        Y = torch.from_numpy(Y).long().to(device)
        X = X.unsqueeze(1)
        # XS = XS.unsqueeze(1)
        output = myNet(X)

        y_pred.extend(torch.argmax(output.detach(), 1).cpu().numpy().tolist())
        y_true.extend(Y.tolist())
        indices.extend(ind.tolist())

    indices=np.array(indices)
    labels_show=np.zeros(mydata.image_shape[0:2])
    labels_show[indices[:,0],indices[:,1]]=np.array(y_pred)+1

    print(metrics.accuracy_score(y_true,y_pred))
    print_metrics(y_true,y_pred)
    plt.imshow(labels_show, cmap='jet')
    plt.axis('off')
    w,h=labels_show.shape
    plt.gcf().set_size_inches(h / 100.0 / 3.0, w / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0,0)
    figname = modelname.split('.')[0] + "_single.png"
    plt.savefig(figname,dpi=600)

    plt.subplot(121)
    plt.axis('off')
    plt.margins(0,0)
    # plt.imshow(mydata.labels, cmap='tab20')
    plt.imshow(mydata.labels, cmap='jet')
    plt.title('true labels')
    plt.subplot(122)
    plt.axis('off')
    plt.margins(0,0)

    plt.imshow(labels_show, cmap='jet')
    plt.title('predict labels')
    # plt.imshow(labels_show, cmap='tab20')
    figname=modelname.split('.')[0]+".png"
    plt.savefig(figname,dpi=600)
    plt.show()
    pass

def test_val(mydata,myNet,fortype='test'):
    print(fortype,'...')
    myNet.eval()
    # device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    y_pred=[]
    y_true=[]
    iters=mydata.test_iters if fortype=='test' else mydata.val_iters
    for iter in range(iters):
        # X, XS, Y = mydata.get_batch_twoimage(iter,fortype)
        X ,Y,ind = mydata.get_batch_oneimage(iter,fortype=fortype)
        X = np.transpose(X, [0, 3, 1, 2])
        X = torch.from_numpy(X).requires_grad_(True).to(device)

        Y = torch.from_numpy(Y).long().to(device)
        X = X.unsqueeze(1)
        output = myNet(X)

        y_pred.extend(torch.argmax(output.detach(),1).cpu().numpy().tolist())
        y_true.extend(Y.tolist())
    acc=metrics.accuracy_score(y_true,y_pred)
    return acc,y_pred,y_true


def run(args=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # if(torch.cuda.is_available()):
    #     device='cuda:0'
    # else:
    #     device='cpu'
    # device='cpu'
    # args=Args()

    print("args dict:", args.__dict__)
    # load data
    image_raw, labels_raw = load_data(args.data_name, args.base_nC)
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

    image_pca = image_raw
    # seg_img_pca = seg_img
    image_pca = preprocess(image_pca)
    # seg_img_pca = preprocess(seg_img_pca, 1)  # 2 0.9138

    mydata = MyData(image_pca, labels_raw, train_ind, test_ind, val_ind, args.patch_size, args.batch_size)
    # test_and_show(mydata)
    # prepare cnnNet
    # myNet=MyNet(args.classnums,args.patch_size,args.batch_size)
    # myNet.to(device)
    # todo one or two image :net
    bands_num = image_pca.shape[-1]
    # bands_seg_num = seg_img_pca.shape[-1]
    myNet = MyNet(args.classnums, bands_num, args.patch_size, args.batch_size)
    myNet.to(device)

    # modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    # myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    # test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    # print('test_acc:', test_acc)
    # print_metrics(y_true,y_pred)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(myNet.parameters(), lr=0.0005, momentum=0.9,weight_decay=1e-4)
    # optim = torch.optim.Adagrad(myNet.parameters(), lr=0.0005, weight_decay=0.01)

    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in myNet.state_dict():
        print(param_tensor, "\t", myNet.state_dict()[param_tensor].size())
    # print("optimizer's state_dict:")
    # # Print optimizer's state_dict
    # for var_name in optim.state_dict():
    #     print(var_name, "\t", optim.state_dict()[var_name])
    val_bst = 0
    bst_epoch = 0
    loss_min = 1e5
    loss_bst_epoch = 0
    for epoch in range(args.epochs):
        # cycle for train,train a net between image and segimage
        # myOneNet.train()
        myNet.train()
        mydata.shuffle()
        for iter in range(mydata.train_iters):
            optim.zero_grad()
            # todo one or two image : train
            # two image,image and segimage
            X, Y ,inds= mydata.get_batch_oneimage(iter)
            # X, XS, Y = mydata.get_batch_oneimage(iter,imagename='seg',fortype='train')
            X = np.transpose(X, [0, 3, 1, 2])
            # XS = np.transpose(XS, [0, 3, 1, 2])
            X = torch.from_numpy(X).requires_grad_(True).to(device)
            # XS = torch.from_numpy(XS).requires_grad_(True).to(device)
            Y = torch.from_numpy(Y).long().to(device)
            X = X.unsqueeze(1)
            # XS = XS.unsqueeze(1)
            output = myNet(X)

            loss = criterion(output, Y)
            loss.backward()
            optim.step()
            print('\repoch:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.train_iters,
                                                                loss.detach().cpu()), end='')

        cur_loss = loss.detach().cpu()
        print('\repoch:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.train_iters,
                                                            cur_loss))

        if (epoch + 1) % 5 == 0:
            # save best model in val_set
            val_acc, _, _ = test_val(mydata, myNet, 'val')
            print('val_acc:', val_acc)
            if epoch - bst_epoch > args.early_stop_epoch:
                print('train break since val_acc does not improve from epoch: ', bst_epoch)
                break
            if val_acc >= val_bst:
                val_bst = val_acc
                bst_epoch = epoch + 1
                modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
                torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
                print("save model:", modelname)
            if epoch - loss_bst_epoch > args.early_stop_epoch:
                print('train break since loss does not decrease from epoch: ', loss_bst_epoch)
                break
            if cur_loss <= loss_min:
                loss_min = cur_loss
                loss_bst_epoch = epoch + 1
                modelname = args.method_name + '_' + args.data_name + '_lossbst' + '.model'
                torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
                print("save model:", modelname)

    # test on best model
    modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    test_acc, y_pred, y_true = test_val(mydata, myNet, 'test')
    print('test_acc:', test_acc)
    # modelname = args.method_name + '_' + args.data_name + '_last' + '.model'
    # torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))
    # if (epoch+1)%5==0:
    #     val_acc, _, _ = test_val(mydata, myNet, 'val')
    #     print('val_acc:', val_acc)
    #     # modelname=args.data_name+str(epoch)+'valacc'+str(val_acc)+'.model'
    #     modelname=args.data_name+'_two_image.model'
    #     torch.save(myNet.state_dict(),os.path.join(args.model_save_path,modelname))
    # test_acc,y_pred,y_true = test_val(mydata, myNet, 'test')
    # print('test_acc:', test_acc)

    overall_accuracy, average_accuracy, kp, classify_report, confusion_matrix, acc_for_each_class = print_metrics(
        y_true, y_pred)

    a, b = np.unique(y_true, return_counts=True)
    print('y_true cnt:', list(zip(a, b)))
    a, b = np.unique(y_pred, return_counts=True)
    print('y_pred cnt:', list(zip(a, b)))
    return overall_accuracy, average_accuracy, kp, acc_for_each_class


def run_more(args):
    global  seed
    oas = []
    aas = []
    kps = []
    afes = []
    for run_id in range(5):
        # seed=run_id+2000
        seed = np.random.randint(10000)
        oa, aa, kp, acc_for_each_class = run(args)
        print('run_id:', run_id)
        print('oa:{} aa:{} kp:{}'.format(oa, aa, kp))
        oas.append(oa)
        aas.append(aa)
        kps.append(kp)
        afes.append(acc_for_each_class)
    print('afes', afes)
    print('oas:', oas)
    print('aas:', aas)
    print('kps', kps)
    print('mean and std:oa,aa,kp')
    print(np.mean(oas) * 100, np.mean(aas) * 100, np.mean(kps) * 100)
    print(np.std(oas), np.std(aas), np.std(kps))
    print('mean/std oa for each class,axis=0:')
    m_afes = np.mean(afes, axis=0)
    for a in m_afes:
        print(a * 100)
    print(np.mean(oas) * 100)
    print(np.mean(aas) * 100)
    print(np.mean(kps) * 100)
    print(np.std(oas) * 100)
    print(np.std(aas) * 100)
    print(np.std(kps) * 100)
    print(np.std(afes, axis=0))
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


class Args():
    def __init__(self, data_name='Salinas'):
        self.method_name = 'SSRN'
        #
        self.data_name = data_name
        self.set_dataset(self.data_name)
        # self.batch_size=128
        self.batch_size = 16
        self.epochs = 200

        self.model_save_path = '.\\models'
        self.early_stop_epoch = 30
        self.record = True
        self.recordFile = self.method_name+'_'+self.data_name+'_results.txt'

    def set_dataset(self, data_name):
        if data_name == 'PaviaU':
            self.data_name = 'PaviaU'
            self.train_size = 50
            self.classnums = 9
            self.base_nC = 50
            self.patch_size = 7
            self.save_path=PaviaU_path
        elif data_name == 'IndianPines':
            self.data_name = 'IndianPines'
            self.train_size = 50
            self.classnums = 16
            self.base_nC = 50
            self.patch_size = 5
            self.save_path = IndianPine_path
        elif data_name == 'Salinas':
            self.data_name = 'Salinas'
            self.train_size = 5
            self.classnums = 16
            self.base_nC = 50
            self.patch_size = 5
            self.save_path = Salinas_path
        elif data_name == 'Houston2013':
            self.data_name = 'Houston2013'
            self.train_size = 50
            self.classnums = 15
            self.base_nC = 200
            self.patch_size = 7
            self.save_path = Houston2013_path


if __name__ == '__main__':
    # test_and_show()
    # exit()
    # run()
    args=Args()
    # args.set_dataset('PaviaU')
    # args.set_dataset('IndianPines')
    args.set_dataset('Houston2013')
    # args.set_dataset('Salinas')
    image_raw, labels_raw = load_data(args.data_name, args.base_nC)
    w,h,b=image_raw.shape
    results=np.zeros_like(image_raw)
    for numSegments in [30,50,70,100,200]:
        print('process seg num:',numSegments)
        for i in tqdm(range(b)):
            image=image_raw[:,:,i]
            segments = slic(image, n_segments=numSegments, sigma=5)
            results[:,:,i]=segments
        save_name=os.path.join(args.save_path,args.data_name+'_SLIC'+str(numSegments)+'.mat')
        sio.savemat(save_name,{'results':results})
        print('saved ',save_name)
    # show the plots
    exit()






    args.patch_size = 25
    test_and_show(args)
    exit()

    args=Args()
    # args.set_dataset('Houston2013')
    # args.set_dataset('Houston2013')
    args.set_dataset('IndianPines')
    # args.record=False
    args.patch_size=25
    args.train_size=50
    # run(args)
    # test_and_show(args)
    run_more(args)
    exit()


    # args.set_dataset('PaviaU')
    # args.set_dataset('Salinas')
    # args.set_dataset('Houston2013')

    patch_grid=[3,5,7,9,11]
    nC_grid=[30,50,70,100,200]
    for patch in patch_grid:
        for nC in nC_grid:
            args = Args()
            # args.set_dataset('IndianPines')
            # args.set_dataset('PaviaU')
            # args.set_dataset('Salinas')
            args.set_dataset('Houston2013')
            args.patch_size = patch
            args.base_nC=nC
            args.method_name=args.method_name+'p{}_nC{}'.format(args.patch_size,args.base_nC)
            print('running: patch:{} nC:{}'.format(args.patch_size,args.base_nC))
            run_more(args)




