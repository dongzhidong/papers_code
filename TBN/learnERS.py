#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 17:08
# @Author  : dongdong
# @FileName: ERSsegment.py
# @Software: PyCharm

from ERSModule import *
from sklearn.decomposition import  PCA
import  numpy as np
import  cv2
import scipy.io as sio
import  os
# import copy.c
import matplotlib.pyplot as plt
Datasets_path=r'.\Datasets'
IndianPine_path=os.path.join(Datasets_path,'IndianPines')
PaviaU_path=os.path.join(Datasets_path,'PaviaU')

data_num=0
datasets=['IndianPines','PaviaU','Salinas','Houston2013']
dataset_name=datasets[data_num]

def load_data(dataset_name):
    if dataset_name=='IndianPines':
        image = sio.loadmat(os.path.join(IndianPine_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(IndianPine_path, 'labels.mat'))['labels'].reshape(145, 145).T
    elif dataset_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
    # elif dataset_name=='Salinas':
    #
    # elif dataset_name=='Houston2013':

    else:
        print('dataset name is error...')
        exit()
    return image,labels

if __name__=='__main__':
    image,labels=load_data(dataset_name)
    h,w,b=image.shape
    nC=50
    conn8=1
    lamb=0.5
    sigma=5.0
    # img=np.reshape(image,[h*w,b])
    # img=PCA(n_components=1).fit_transform(img)
    img=copy(np.array(image[:,:,0],dtype=np.float))
    # mmin=np.min(img)
    # mmax=np.max(img)
    # img=np.ceil((img-mmin)/(mmax-mmin)*255)
    # img_show=img

    img=np.array(img,dtype=np.float)
    img=np.ravel(img).tolist()
    # Arg: img_list, h, w, nC, conn8=0, lambda=0.5, sigma=5.0
    label_list=ERS(img,h,w,nC,conn8,lamb,sigma)
    # label_list=ERS(img,h,w,nC)
    label=np.reshape(np.asarray(label_list), (h, w))
    img_show = np.reshape(img, (h, w))
    plt.subplot(1,2,1)
    plt.imshow(img_show,'jet')
    plt.subplot(1,2,2)
    plt.imshow(label,'jet')
    plt.show()
    print('end')



