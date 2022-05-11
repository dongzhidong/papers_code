import matplotlib.pyplot as plt
import  numpy as np
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
Dataset_path='.\\Datasets'
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')
def load_data(data_name,nC=50,show=False):
    # seg_name = data_name + '_255seg' + str(nC) + '.mat'
    seg_name = data_name + '_SLIC' + str(nC) + '.mat'
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
        # Seg_img=sio.loadmat(os.path.join(IndianPine_path,'Indian_seg.mat'))['seg_labels']
        Seg_img=sio.loadmat(os.path.join(IndianPine_path,seg_name))['results']
        # train_rate=0.05
        channel_show=10
        # seg_pca1=sio.loadmat(os.path.join(IndianPine_path,'IndianPines_pca50.mat'))['labels']
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        Seg_img=sio.loadmat(os.path.join(PaviaU_path,seg_name))['results']
        # train_rate=0.03
        channel_show=100
        # seg_pca1=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_pca50.mat'))['labels']
    elif data_name=='Salinas':
        image = sio.loadmat(os.path.join(Salinas_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(Salinas_path, 'Salinas_gt.mat'))['salinas_gt']
        Seg_img = sio.loadmat(os.path.join(Salinas_path, seg_name))['results']
        channel_show = 10

    elif data_name=='Houston2013':
        # seg_name='Houston2013_255seg'+str(nC)+'.mat'
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']
        Seg_img = sio.loadmat(os.path.join(Houston2013_path, seg_name))['results']
        channel_show = 10

    Seg_img=Seg_img+1
    if show:
        plt.subplot(131)
        plt.title('raw data '+str(channel_show)+'th channel')
        plt.imshow(image[:,:,channel_show])
        plt.subplot(132)
        plt.title('true labels')
        plt.imshow(labels)
        plt.subplot(133)
        plt.title('Seg data '+str(channel_show)+'th channel')
        plt.imshow(Seg_img[:,:,channel_show])
        plt.show()
    return np.array(image,dtype=np.float),np.array(labels,dtype=np.int),np.array(Seg_img,dtype=np.int)

class Args():
    def __init__(self, data_name='Salinas'):
        self.method_name = 'twoBranch_'
        #
        self.data_name = data_name
        self.set_dataset(self.data_name)
        # self.batch_size=128
        self.batch_size = 16
        self.epochs = 200

        self.model_save_path = '.\\models'
        self.early_stop_epoch = 30
        self.record = True
        self.recordFile = 'twoBranchPavia_results.txt'

    def set_dataset(self, data_name):
        if data_name == 'PaviaU':
            self.data_name = 'PaviaU'
            self.train_size = 50
            self.classnums = 9
            self.base_nC = 50
            self.patch_size = 7
        elif data_name == 'IndianPines':
            self.data_name = 'IndianPines'
            self.train_size = 50
            self.classnums = 16
            self.base_nC = 50
            self.patch_size = 5
        elif data_name == 'Salinas':
            self.data_name = 'Salinas'
            self.train_size = 5
            self.classnums = 16
            self.base_nC = 50
            self.patch_size = 5
        elif data_name == 'Houston2013':
            self.data_name = 'Houston2013'
            self.train_size = 50
            self.classnums = 15
            self.base_nC = 200
            self.patch_size = 7

args=Args()
args.set_dataset('PaviaU')
image_raw, labels_raw, seg_img = load_data(args.data_name, args.base_nC)
# seg=seg_img[:,:,-1]
# plt.imshow(seg, cmap='jet')
plt.imshow(labels_raw,cmap='jet')
plt.axis('off')
h,w=labels_raw.shape
plt.gcf().set_size_inches(w/100.0/3.0,h/100.0/3.0)
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.margins(0,0)
# plt.show()
# figname = modelname.split('.')[0] + "_labels.png"
figname = "PU_labels.png"
plt.savefig(figname,dpi=600)

print('end')