from paramters import *
from scipy import io as sio
import numpy as np
from tools import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import os
from joblib import load,dump
Dataset_path='.\\Datasets'
IndianPine_path=os.path.join(Dataset_path,'IndianPines')
PaviaU_path=os.path.join(Dataset_path,'PaviaU')
Salinas_path=os.path.join(Dataset_path,'Salinas')
Houston2013_path=os.path.join(Dataset_path,'Houston2013')

def preprocess(dataset, normalization = 2):
    '''
    对数据集进行归一化；
    normalization = 1 : 0-1归一化；  2：标准化； 3：分层标准化；4.逐点正则化
    '''
    #attation 数据归一化要做在划分训练样本之前；
    dataset = np.array(dataset, dtype = 'float64')
    [m,n,b] = np.shape(dataset)
    #先进行数据标准化再去除背景  效果更好
    if normalization==0:
        return (dataset)
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
    if data_name=='IndianPines':
        image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
        # Seg_img=sio.loadmat(os.path.join(IndianPine_path,'Indian_seg.mat'))['seg_labels']
        Seg_img=sio.loadmat(os.path.join(IndianPine_path,'IndianPines_seg50.mat'))['results']
        # train_rate=0.05
        channel_show=10
        # seg_pca1=sio.loadmat(os.path.join(IndianPine_path,'IndianPines_pca50.mat'))['labels']
    elif data_name=='PaviaU':
        image=sio.loadmat(os.path.join(PaviaU_path,'PaviaU.mat'))['paviaU']
        labels=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_gt.mat'))['paviaU_gt']
        Seg_img=sio.loadmat(os.path.join(PaviaU_path,'Pavia_seg.mat'))['results']
        # train_rate=0.03
        channel_show=100
        # seg_pca1=sio.loadmat(os.path.join(PaviaU_path,'PaviaU_pca50.mat'))['labels']
    elif data_name=='Salinas':
        image = sio.loadmat(os.path.join(Salinas_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(Salinas_path, 'Salinas_gt.mat'))['salinas_gt']
        Seg_img = sio.loadmat(os.path.join(Salinas_path, 'Salinas_seg50.mat'))['results']
        channel_show = 10

    elif data_name=='Houston2013':
        seg_name='Houston2013_255seg'+str(nC)+'.mat'
        image = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013.mat'))['Houston']
        labels = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_gt.mat'))['labels']
        Seg_img = sio.loadmat(os.path.join(Houston2013_path, 'Houston2013_255seg100.mat'))['results']
        channel_show = 10
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
    return data_name,np.array(image,dtype=np.float),np.array(Seg_img,dtype=np.int),np.array(labels,dtype=np.int)

def test_and_show(data_name,image,Seg_img,labels,train_rate,featuretype='pcaraw+pcaseg',normalization=2,random_state=None,show=False):
    # load data
    w, h, b = image.shape
    save_name = "SVM_" + data_name + "_" + featuretype + ".joblib"
    if featuretype == 'pcaraw+pcaseg':
        if data_name == 'IndianPines':
            data = np.concatenate((my_pca(image, 0.9), my_pca(Seg_img)), 2)
        else:
            data = np.concatenate((my_pca(image, 0.9), my_pca(Seg_img)), 2)
    elif featuretype == 'pcaseg':
        data = my_pca(Seg_img)
    elif featuretype == 'pcaraw':
        data = my_pca(image)
    elif featuretype == 'seg':
        data = Seg_img
    elif featuretype == 'raw':
        data = image
    elif featuretype == 'raw+seg' or featuretype == 'seg+raw':
        data = np.concatenate((Seg_img, image), 2)
    elif featuretype == 'raw+pcaseg' or featuretype == 'pcaseg+raw':
        data = np.concatenate((my_pca(Seg_img), image), 2)
    # elif featuretype=='rawbypca1':
    #     data=reduce_by_pca1(image,seg_pca1)
    # elif featuretype == 'pcaraw+pca1':
    #     data = np.concatenate((my_pca(image), seg_pca1[:, :, np.newaxis]), 2)

    data = preprocess(data, normalization)
    print('-------------------------------')
    print('dataset:{} featuretype:{} normalization:{} '.format(data_name, featuretype, normalization))
    print(data.shape)
    remain_index = np.argwhere(labels != 0)
    # remain_index = np.array(list(zip(remain_index[0], remain_index[1])))
    testX=data[remain_index[:,0],remain_index[:,1],:]
    testY=labels[remain_index[:,0],remain_index[:,1]]

    modelname = save_name
    print('load model:',modelname)
    clf=load(modelname)
    print('test all...')
    y_pred = clf.predict(testX)

    labels_show=np.zeros(image.shape[0:2])
    labels_show[remain_index[:,0],remain_index[:,1]]=y_pred

    print(metrics.accuracy_score(testY,y_pred))
    print_metrics(testY,y_pred)
    plt.imshow(labels_show,cmap='jet')
    plt.axis('off')
    plt.margins(0,0)
    figname = modelname.split('.')[0] + "_single.png"
    plt.savefig(figname,dpi=600)

    plt.subplot(121)
    plt.axis('off')
    plt.margins(0,0)
    # plt.imshow(mydata.labels, cmap='tab20')
    plt.imshow(labels, cmap='jet')
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


def run_svm(data_name,image,Seg_img,labels,train_rate,featuretype='pcaraw+pcaseg',normalization=2,random_state=None,show=False):
    w,h,b=image.shape
    save_name="SVM_"+data_name+"_"+featuretype+".joblib"
    if featuretype=='pcaraw+pcaseg':
        if data_name=='IndianPines':
            data = np.concatenate((my_pca(image, 0.9), my_pca(Seg_img)), 2)
        else:
            data=np.concatenate((my_pca(image,0.9),my_pca(Seg_img)),2)
    elif featuretype=='pcaseg':
        data=my_pca(Seg_img)
    elif featuretype == 'pcaraw':
        data = my_pca(image)
    elif featuretype=='seg':
        data=Seg_img
    elif featuretype == 'raw':
        data = image
    elif featuretype=='raw+seg' or  featuretype=='seg+raw' :
        data=np.concatenate((Seg_img,image),2)
    elif featuretype=='raw+pcaseg' or  featuretype=='pcaseg+raw' :
        data=np.concatenate((my_pca(Seg_img),image),2)
    # elif featuretype=='rawbypca1':
    #     data=reduce_by_pca1(image,seg_pca1)
    # elif featuretype=='pcaraw+pca1':
    #     data=np.concatenate((my_pca(image),seg_pca1[:,:,np.newaxis]),2)

    data=preprocess(data,normalization)
    print('-------------------------------')
    print('dataset:{} featuretype:{} normalization:{} '.format(data_name,featuretype,normalization))
    print(data.shape)

    data=data.reshape([data.shape[0]*data.shape[1],data.shape[2]])
    labels=labels.reshape([labels.shape[0]*labels.shape[1],1])
    trainX, testX, trainY, testY=my_train_test_split(data,labels,train_rate,random_state)
    remain_index=np.argwhere(labels!=0)[:,0]
    # trainX,testX,trainY,testY=train_test_split(data[remain_index],labels[remain_index],train_size=train_rate,random_state=random_state)
    # for it in np.unique(trainY):
    #     print('*{}: {}'.format(it, np.argwhere(testY == it)[:, 0].shape[0]))
    # trainX,testX,trainY,testY=train_test_split(data[remain_index],labels[remain_index],train_size=train_rate)
    print('训练数据：',trainX.shape,' 总可用样本：',remain_index.shape)
    testY=testY.ravel()
    trainY=trainY.ravel()

    # clf=SVC(decision_function_shape='ovr',class_weight='balanced')
    clf = SVC(decision_function_shape='ovr', C=100, gamma=0.01)
    clf.fit(trainX,trainY)
    dump(clf,save_name)
    y_pred=clf.predict(testX)
    oa,aa,kappa,classify_report,confusion_matrix,acc_for_each_class=print_metrics(testY,y_pred,show=True)
    print('len train',len(trainY),'  len test: ',len(testY),'  total: ',len(remain_index),' rate: ',len(trainY)/len(remain_index))

    if show:
        pred_labels=clf.predict(data)
        all_pred_labels=pred_labels.copy()
        all_pred_labels[0]=0 #为了与其他保持颜色一致
        drop_index=np.argwhere(labels==0)
        pred_labels[drop_index]=0
        pred_img=pred_labels.reshape([w,h])
        plt.subplot(131)
        plt.imshow(pred_img)
        plt.title('predict labels')
        plt.subplot(132)
        plt.imshow(labels.ravel().reshape([w,h]))
        plt.title('true labels')
        plt.subplot(133)
        plt.imshow(all_pred_labels.reshape([w,h]))
        plt.title('predict all pixels labels')
        plt.show()
    return oa,aa,kappa,classify_report,confusion_matrix,acc_for_each_class


if __name__ == '__main__':
    train_rate=5
    data_name='IndianPines'
    # data_name='PaviaU'
    # data_name='Salinas'
    # data_name='Houston2013'

    normalization=2
    # featuretype='pcaraw+pcaseg'
    # featuretype='pcaraw'
    featuretype='raw'
    # featuretype='pcaseg'
    # featuretype='raw+pcaseg'
    # featuretype='pcaraw+pca1'
    # featuretype='rawbypca1'

    random_state=None
    run_nums=1
    data_name, image, Seg_img, labels=load_data(data_name)
    f=open('svm_results0108.txt','a')
    f.write('\n***********************************\n')
    f.write(time.asctime())
    f.write('\n-------------------')
    f.write('\ndata:{} featuretype:{}  train_rate:{}  normalization:{} run_nums:{} random_state:{}'.format(
        data_name,featuretype,train_rate ,normalization,run_nums,random_state
    ))
    f.flush()
    oas=[]
    aas=[]
    kps=[]
    eca_list=[]
    start_time=time.clock()
    for run_id in range(run_nums):
        print('---------------------------------')
        print('run_id:{}/{}'.format(run_id,run_nums))
        print('---------------------------------')
        # oa, aa, kappa, classify_report, confusion_matrix, acc_for_each_class\
        #     =run_svm(data_name, image, Seg_img, labels,  train_rate,featuretype,normalization,random_state=random_state)
        test_and_show(data_name, image, Seg_img, labels,  train_rate,featuretype,normalization,random_state=random_state)

        oas.append(oa)
        aas.append(aa)
        kps.append(kappa)
        eca_list.append(acc_for_each_class)
        # f.write('\nrunid:{}/{} oa:{} aa:{} kappa:{}'.format(run_id,run_nums,oa,aa,kappa) )
        # f.write(classify_report)
        # f.write(confusion_matrix)
        # f.write(acc_for_each_class)
        # f.write('oa:{} aa:{} kappa:{}'.format(oa, aa, kappa))
        # f.flush()
    # f.write('\ndata:{} featuretype:{} normalization:{} run_nums:{} random_state{}'.format(
    #     data_name, featuretype, normalization, run_nums, random_state
    # ))
    print('总用时：',time.clock()-start_time)
    f.write('\n平均值：oa：{}（{}） aa:{}（{}） kappa:{}（{}）'.format(str(np.mean(oas)),np.std(oas),str(np.mean(aas)),
                                                            np.std(aas),str(np.mean(kps)),np.std(kps)))
    f.close()
    print('\n平均值：oa：{}（{}） aa:{}（{}） kappa:{}（{}）'.format(str(np.mean(oas)),np.std(oas),str(np.mean(aas)),
                                                            np.std(aas),str(np.mean(kps)),np.std(kps)))
    eca_mean=np.mean(np.array(eca_list),0)
    eca_std=np.std(np.array(eca_list),0)
    print('\n各类均值：')
    for i in range(len(eca_list[0])):
        print(i+1,': {}({})'.format(eca_mean[i],eca_std[i]))
    print('end')
