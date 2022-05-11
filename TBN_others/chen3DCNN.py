#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 2021/5/15 13:57
# @Author  : dongdong
# @File    : Hamida3DCNN.py
# @Software: PyCharm
'''
Hamida3DCNN
MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL IMAGE CLASSIFICATION


'''
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
from sklearn import metrics,preprocessing

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
# import torch.optim.optimizer as optimizer
Dataset_path='.\\Datasets'
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')
device='cuda:0' if torch.cuda.is_available() else 'cpu'
seed=2021
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusion_matrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('-----------------------------------------')
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:',kp)
    return overall_accuracy,average_accuracy,kp,classify_report,confusion_matrix,acc_for_each_class

def load_data(data_name,show=False):
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
        channel_show=10
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        channel_show=100
    elif data_name=='Salinas':
        image=sio.loadmat(os.path.join(Salinas_path,'Salinas_corrected.mat'))['salinas_corrected']
        labels=sio.loadmat(os.path.join(Salinas_path,'Salinas_gt.mat'))['salinas_gt']
        channel_show = 10
    elif data_name=='Houston2013':
        # seg_name='Houston2013_255seg'+str(nC)+'.mat'
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']
        # Seg_img = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_255seg100.mat'))['results']
        channel_show = 10

    if show:
        plt.subplot(121)
        plt.title('raw data '+str(channel_show)+'th channel')
        plt.imshow(image[:,:,channel_show])
        plt.subplot(122)
        plt.title('true labels')
        plt.imshow(labels)
        # plt.subplot(133)
        # plt.title('Seg data '+str(channel_show)+'th channel')
        # plt.imshow(Seg_img[:,:,channel_show])
        plt.show()
    return np.array(image,dtype=np.float),np.array(labels,dtype=np.int)
class MyData():
    def __init__(self,image,labels,train_ind,test_ind,val_ind=[],patch_size=1,batch_size=128):
        w,h,b=image.shape
        self.batch_size=batch_size
        self.patch_size=patch_size
        self.channels=image.shape[-1]
        self.train_ind=train_ind
        self.test_ind=test_ind
        self.val_ind=val_ind
        self.labels=labels
        self.image_pad=np.zeros((w+patch_size-1,h+patch_size-1,b))
        self.image_pad[patch_size//2:w+patch_size//2,patch_size//2:h+patch_size//2,:]=image

        np.random.shuffle(self.train_ind)
        self.train_iters=len(self.train_ind)//batch_size+1
        self.test_iters = len(self.test_ind) // batch_size+1
        self.val_iters=len(self.val_ind)//batch_size+1

    def get_batch(self,iter):
        '''return the patches of 'iter' batch in page_i of image'''
        indices=self.train_ind[iter*self.batch_size:(iter+1)*self.batch_size]
        X=np.zeros((self.batch_size,self.patch_size,self.patch_size,self.channels))
        Y=np.zeros((self.batch_size,))
        hp=self.patch_size
        for ind_i,ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i,:,:,:]=self.image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            Y[ind_i]=self.labels[ind[0],ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1

    def get_batch_oneimage(self,iter,fortype='train'):
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
        Y=np.zeros((self.batch_size,))
        hp=self.patch_size

        for ind_i,ind in enumerate(indices):
            # Pay attention to coordinate changes
            X[ind_i,:,:,:]=self.image_pad[ind[0]:ind[0]+hp,ind[1]:ind[1]+hp,:]
            Y[ind_i]=self.labels[ind[0],ind[1]]
        return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float)-1

    def shuffle(self):
        np.random.shuffle(self.train_ind)
class Chen3DCNN(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    # (args.classnums, bands_num, args.patch_size, args.batch_size)
    def __init__(self, n_classes,bands_num, patch_size=27, batch_size=32):
        super(Chen3DCNN, self).__init__()
        self.input_channels = bands_num
        n_planes=32
        self.n_planes = 32
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


# class Hamida3DCNN(nn.Module):
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
#             init.kaiming_uniform(m.weight)
#             init.zeros_(m.bias)
#
#     def __init__(self,classnums,bands_num,patch_size=1,batch_size=128):
#         super(Hamida3DCNN, self).__init__()
#         # The first layer is a (3,3,3) kernel sized Conv characterized
#         # by a stride equal to 1 and number of neurons equal to 20
#         self.patch_size = patch_size
#         self.input_channels = bands_num
#         dilation = (1, 1, 1)
#
#         if patch_size == 3:
#             self.conv1 = nn.Conv3d(
#                 1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
#         else:
#             self.conv1 = nn.Conv3d(
#                 1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 0, 0))
#         # Next pooling is applied using a layer identical to the previous one
#         # with the difference of a 1D kernel size (1,1,3) and a larger stride
#         # equal to 2 in order to reduce the spectral dimension
#         self.conv2 = nn.Conv3d(
#             20, 2, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
#         # Then, a duplicate of the first and second layers is created with
#         # 35 hidden neurons per layer.
#         self.conv3 = nn.Conv3d(
#             2, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
#         self.conv4 = nn.Conv3d(
#             35, 2, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
#         # Finally, the 1D spatial dimension is progressively reduced
#         # thanks to the use of two Conv layers, 35 neurons each,
#         # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
#         # respectively equal to (1,1,1) and (1,1,2)
#
#
#         self.conv5 = nn.Conv3d(
#             2, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
#         self.conv6 = nn.Conv3d(
#             35, 2, (1, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
#         self.conv7 = nn.Conv3d(
#             2, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
#         self.conv8 = nn.Conv3d(
#             35, 4, (1, 1, 1), dilation=dilation, stride=(2, 2, 2))
#
#         self.dropout = nn.Dropout(p=0.5)
#
#         self.features_size = self._get_final_flattened_size()
#         # The architecture ends with a fully connected layer where the number
#         # of neurons is equal to the number of input classes.
#         self.fc = nn.Linear(self.features_size, classnums)
#
#         self.apply(self.weight_init)
#
#     def _get_final_flattened_size(self):
#         with torch.no_grad():
#             x = torch.zeros((1, 1, self.input_channels,
#                              self.patch_size, self.patch_size))
#             x = self.conv1(x)
#             x = self.conv2(x)
#             x = self.conv3(x)
#             x = self.conv4(x)
#             # x = self.conv5(x)
#             # x = self.conv6(x)
#             # x = self.conv7(x)
#             # x = self.conv8(x)
#             _, t, c, w, h = x.size()
#         return t * c * w * h
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         # x = F.relu(self.conv5(x))
#         # x = F.relu(self.conv6(x))
#         # x = F.relu(self.conv7(x))
#         # x = F.relu(self.conv8(x))
#         x = x.view(-1, self.features_size)
#         # x = self.dropout(x)
#         x = self.fc(x)
#         return x
#


def test_val(mydata,myNet,fortype='test'):
    print(fortype,'...')
    myNet.eval()

    y_pred=[]
    y_true=[]
    iters=mydata.test_iters if fortype=='test' else mydata.val_iters
    for iter in range(iters):
        X, Y = mydata.get_batch_oneimage(iter,fortype)
        X=np.transpose(X,[0,3,1,2])
        X = torch.from_numpy(X).requires_grad_(True).to(device)
        X = X.unsqueeze(1)
        output = myNet(X)
        y_pred.extend(torch.argmax(output.detach(),1).cpu().numpy().tolist())
        y_true.extend(Y.tolist())
    acc=metrics.accuracy_score(y_true,y_pred)
    return acc,y_pred,y_true


def run():
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # if(torch.cuda.is_available()):
    #     device='cuda:0'
    # else:
    #     device='cpu'
    args=Args()
    print("args dict:",args.__dict__)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # load data
    image_raw,labels_raw=load_data(args.data_name)
    remain_index=np.where(labels_raw!=0)
    remain_index=np.array(list(zip(remain_index[0],remain_index[1])))
    train_ind,test_ind,_,_=my_train_test_split(remain_index,labels_raw[remain_index[:,0],remain_index[:,1]],args.train_size,random_state=2021)
    val_ind, _ = train_test_split(test_ind, train_size=0.2, random_state=2021)

    print('trian samples:',len(train_ind))
    a,b=np.unique(labels_raw[train_ind[:,0],train_ind[:,1]],return_counts=True)
    print(list(zip(a,b)))
    print('test samples:')
    a,b=np.unique(labels_raw[test_ind[:,0],test_ind[:,1]],return_counts=True)
    print(list(zip(a,b)))

    # image_pca=my_pca(image_raw,30)
    image_pca = image_raw
    image_pca=preprocess(image_pca)
    # todo here i replace image_pca for convinience,it should be rewrite
    # image_pca=seg_img_pca

    # seg_lable_cnt=len(np.unique(seg_img))
    mydata = MyData(image_pca,labels_raw,train_ind,test_ind,val_ind, args.patch_size, args.batch_size)
    # prepare cnnNet
    # myNet=MyNet(args.classnums,args.patch_size,args.batch_size)
    # myNet.to(device)
    # todo one or two image :net
    bands_num=image_pca.shape[-1]
    myNet = Chen3DCNN(args.classnums,bands_num, args.patch_size, args.batch_size)
    myNet.to(device)

    criterion=nn.CrossEntropyLoss()
    optim=torch.optim.SGD(myNet.parameters(), lr=args.lr, momentum=0.9)
    # optim=torch.optim.Adagrad(myNet.parameters(), lr=args.lr, weight_decay=0.01)
    # optim=torch.optim.Adam(myNet.parameters(), lr=args.lr, weight_decay=0.0005)

    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in myNet.state_dict():
        print(param_tensor, "\t", myNet.state_dict()[param_tensor].size())
    # print("optimizer's state_dict:")
    # # Print optimizer's state_dict
    # for var_name in optim.state_dict():
    #     print(var_name, "\t", optim.state_dict()[var_name])
    val_bst=0
    bst_epoch=0
    for epoch in range(args.epochs):
        # cycle for train,train a net between image and segimage
        myNet.train()
        mydata.shuffle()
        for iter in range(mydata.train_iters):
            optim.zero_grad()
            # one image,
            X,Y=mydata.get_batch_oneimage(iter)
            X=np.transpose(X,[0,3,1,2])
            X=torch.from_numpy(X).requires_grad_(True).to(device)
            Y=torch.from_numpy(Y).long().to(device)
            X=X.unsqueeze(1)
            output=myNet(X)

            loss=criterion(output,Y)
            loss.backward()
            optim.step()
            print('\repoch:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch,args.epochs,iter,mydata.train_iters,loss.detach().cpu()),end='')

        print('\repoch:{}/{} iter:{}/{} loss:{:.6f}'.format(epoch, args.epochs, iter, mydata.train_iters,
                                                                loss.detach().cpu()))

        if (epoch+1)%5==0:
            # save best model in val_set
            val_acc, _, _ = test_val(mydata, myNet, 'val')
            print('val_acc:', val_acc)
            if epoch-bst_epoch>args.early_stop_epoch:
                print('train break since val_acc does not improve from epoch: ',bst_epoch)
                break
            if val_acc>=val_bst:
                val_bst=val_acc
                bst_epoch=epoch+1
                modelname=args.method_name+'_'+args.data_name+'_valbst'+'.model'
                torch.save(myNet.state_dict(),os.path.join(args.model_save_path,modelname))
    #test on best model
    modelname = args.method_name + '_' + args.data_name + '_valbst' + '.model'
    myNet.load_state_dict(torch.load(os.path.join(args.model_save_path, modelname)))
    test_acc,y_pred,y_true = test_val(mydata, myNet, 'test')
    print('test_acc:', test_acc)
    modelname=args.method_name+'_'+args.data_name+'_last' + '.model'
    torch.save(myNet.state_dict(), os.path.join(args.model_save_path, modelname))

    overall_accuracy, average_accuracy, kp, classify_report, confusion_matrix, acc_for_each_class=print_metrics(y_true,y_pred)

    a,b=np.unique(y_true,return_counts=True)
    print('y_true cnt:',list(zip(a,b)))
    a,b=np.unique(y_pred,return_counts=True)
    print('y_pred cnt:',list(zip(a,b)))

    return overall_accuracy,average_accuracy,kp,acc_for_each_class
class Args():
    def __init__(self):
        self.method_name='chen3DCNN'
        self.lr=0.003
        self.data_name='IndianPines'
        self.train_size=50
        self.classnums=16
        #
        # self.data_name = 'PaviaU'
        # self.train_size=20
        # self.classnums=9

        # self.data_name = 'Salinas'
        # self.train_size=5
        # self.classnums=16

        # self.data_name = 'Houston2013'
        # self.train_size = 50
        # self.classnums = 15
        # self.base_nC = 200

        self.patch_size=27
        self.batch_size=128
        self.epochs=300
        self.early_stop_epoch=30
        self.model_save_path='.\\models'

if __name__ == '__main__':
    run()
    exit()

    oas=[]
    aas=[]
    kps=[]
    afes = []
    for run_id in range(5):
        seed=run_id+2000
        oa,aa,kp,acc_for_each_class=run()
        print('run_id:',run_id)
        print('oa:{} aa:{} kp:{}'.format(oa,aa,kp))
        oas.append(oa)
        aas.append(aa)
        kps.append(kp)
        afes.append(acc_for_each_class)
    print('afes', afes)
    print('oas:', oas)
    print('aas:', aas)
    print('kps', kps)
    print('mean and std:oa,aa,kp')
    print(np.mean(oas), np.mean(aas), np.mean(kps))
    print(np.std(oas), np.std(aas), np.std(kps))
    print('mean/std oa for each class,axis=0:')
    print(np.mean(afes, axis=0))
    print(np.std(afes, axis=0))





