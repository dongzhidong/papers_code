from paramters import *
from scipy import io as sio
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# import seaborn as sns
import itertools
# import spectral
import matplotlib.pyplot as plt
from scipy import io, misc
# import imageio
import os
import re
# import torch
import datetime

# def get_device(ordinal):
#     # Use GPU ?
#     if ordinal < 0:
#         print("Computation on CPU")
#         device = torch.device('cpu')
#     elif torch.cuda.is_available():
#         print("Computation on CUDA GPU device {}".format(ordinal))
#         device = torch.device('cuda:{}'.format(ordinal))
#     else:
#         print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
#         device = torch.device('cpu')
#     return device

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    # elif ext == '.tif' or ext == '.tiff':
    #     # Load TIFF file
    #     return imageio.imread(dataset)
    # elif ext == '.hdr':
    #     img = spectral.open_image(dataset)
    #     return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))
def my_train_test_split(data,labels,train_rate,random_state):
    trainX, testX, trainY, testY=None,None,None,None
    for it in np.unique(labels):
        if it == 0:
            continue
        it_index=np.argwhere(labels==it)[:,0]
        if (train_rate<1.0 and it_index.shape[0]*train_rate<5):
            itrainX, itestX, itrainY, itestY = train_test_split(data[it_index,:], labels[it_index], train_size=5,
                                                                random_state=random_state)
        elif (train_rate>1.0 and it_index.shape[0]<train_rate) :
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
def my_pca(image,**kwargs):
    """对image 的通道 进行pca，并恢复到image"""
    w,h,b=len(image),len(image[0]),len(image[0][0])
    image = np.reshape(image, [w*h, b])
    pca=PCA(n_components=0.99)
    pca.fit(image)
    return  np.reshape(pca.transform(image),[w,h,-1])

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

def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h

def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)

def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)

def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def camel_to_snake(name):
    # 将驼峰命名法(testName) 转为 蛇形命名法(test_name)
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


