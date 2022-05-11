#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 12:31
# @Author  : dongdong
# @FileName: SuperPCA.py
# @Software: PyCharm

from  sklearn.pipeline import  make_pipeline
from scipy import io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import  os

Dataset_path='.\\'
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')

data_names=['IndianPines','PaviaU','Salinas','Houston2013']
data_name=data_names[3]
if data_name=='IndianPines':
    img=sio.loadmat(os.path.join(Dataset_path,'Indian_predict_labels.mat'))['predict_labels']
else:
    img = sio.loadmat(os.path.join(Dataset_path, data_name+'_predict_labels50.mat'))['predict_labels']
labels_show=img
plt.imshow(labels_show, cmap='jet')
plt.axis('off')
w,h=labels_show.shape
plt.gcf().set_size_inches(h / 100.0 / 3.0, w / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0,0)
figname = 'superPCA_'+data_name + "_single.png"
plt.savefig(figname,dpi=600)
plt.show()