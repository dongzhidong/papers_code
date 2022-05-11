#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 10:08
# @Author  : dongdong
# @File    : main_classify.py
# @Software: PyCharm
# @Purpose :this is data preprocess for Multi-task HSIC


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
# 参数设置 命令行参数，若为指定则为默认值
parser=argparse.ArgumentParser()
# 数据相关参数
parser.add_argument('--Dataset_path',action='store',type=str,default='./Datasets',required=False,help='the datasets path')
parser.add_argument('--data_name',action='store',type=str,default='PaviaU',required=False,help='IndianPines|PaviaU|Salinas|Houston2013')
pargs=parser.parse_args()

Dataset_path=pargs.Dataset_path
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')

def load_data(data_name):
    ret=[]
    files=[]
    if data_name=='IndianPines':
        files=[fl for fl in os.listdir(IndianPine_path) if '_255seg' in fl]
        for fl in sorted(files):
            flabspath=os.path.join(IndianPine_path,fl)
            ret.append(sio.loadmat(flabspath)['results'])
    elif data_name=='PaviaU':
        files=[fl for fl in os.listdir(PaviaU_path) if '_255seg' in fl]
        for fl in sorted(files):
            flabspath=os.path.join(PaviaU_path,fl)
            ret.append(sio.loadmat(flabspath)['results'])
    elif data_name=='Salinas':
        files=[fl for fl in os.listdir(Salinas_path) if '_255seg' in fl]
        for fl in sorted(files):
            flabspath=os.path.join(Salinas_path,fl)
            ret.append(sio.loadmat(flabspath)['results'])
    elif data_name=='Houston2013':
        files=[fl for fl in os.listdir(Houston2013_path) if '_255seg' in fl]
        for fl in sorted(files):
            flabspath=os.path.join(Houston2013_path,fl)
            ret.append(sio.loadmat(flabspath)['results'])
    else:
        print('Can not find the dataset!')
    if ret:
        ret=np.array(ret,dtype=np.float)
        n,w,h,b=ret.shape
        ret=np.reshape(np.transpose(ret,[1,2,3,0]),[w,h,b*n])
    return ret,sorted(files)

def precessfun(data_name):   
    # load data ,we get the superpixel segment labels,its shape is [w,h,bands*scales]
    labels,files=load_data(data_name)
    print('the '+str(len(files))+'segment files:',files)
    # according the segment labels, infer the edge similarity 
    w,h,b=labels.shape
    # edgerow[a,b] represents the edge type between pixel_(a,b) and its right pixel
    # edgeabove[a,b] represents  the edge type between pixel_(a,b) and its RightTop pixel
    # edgecol and edge below -> bottom and RightBottom
    edgerow=np.zeros((w,h)) # row,col range:[0,w-1],[0,h-2]
    edgecol=np.zeros((w,h))# row,col range:[0,w-2],[0,h-1]
    edgeabove=np.zeros((w,h))# row,col range:[1,w-1],[0,h-2]
    edgebelow=np.zeros((w,h))# row,col range:[0,w-2],[0,h-2]
    for k in tqdm(range(b)):
        for i in range(w):
            for j in range(h):
                if j!=h-1:
                    edgerow[i,j]+= 1.0/b*(labels[i,j,k]==labels[i,j+1,k])
                if i!=w-1:
                    edgecol[i,j]+= 1.0/b*(labels[i,j,k]==labels[i+1,j,k])
                if i!=0 and j!=h-1:
                    edgeabove[i,j]+= 1.0/b*(labels[i,j,k]==labels[i-1,j+1,k])
                if i!=w-1 and j!=h-1:
                    edgebelow[i,j]+= 1.0/b*(labels[i,j,k]==labels[i+1,j+1,k])

    sio.savemat(data_name+'_edgelabels.mat',{'edgerow':edgerow,\
            'edgecol':edgecol,'edgeabove':edgeabove,'edgebelow':edgebelow,\
                'files':files})

if __name__=='__main__':
    data_name=['IndianPines','PaviaU','Salinas','Houston2013'][2]
    print(data_name)
    precessfun(data_name)
    print('end')
