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

image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
labels=sio.loadmat(os.path.join(IndianPine_path,'labels.mat'))['labels'].reshape(145,145).T
Seg_img=sio.loadmat(os.path.join(IndianPine_path,'Indian_seg.mat'))['seg_labels']
test_rate=0.9
channel_show=10

# image=sio.loadmat(os.path.join(Houston2013_path,'Houston2013.mat'))['Houston']
# labels=sio.loadmat(os.path.join(Houston2013_path,'Houston2013_gt.mat'))['labels']
# Seg_img=sio.loadmat(os.path.join(Houston2013_path,'Houston2013_255seg100.mat'))['results']
# test_rate=0.9
# train_size=50*15
# channel_show=10



# plt.subplot(131)
# plt.title('raw data '+str(channel_show)+'th channel')
# plt.imshow(image[:,:,channel_show])
# plt.subplot(132)
# plt.title('true labels')
# plt.imshow(labels)
# plt.subplot(133)
# plt.title('Seg data '+str(channel_show)+'th channel')
# plt.imshow(Seg_img[:,:,channel_show])
# plt.show()
w,h,b=image.shape


# data=np.concatenate((image,Seg_img),2)
# data=np.concatenate((my_pca(image),my_pca(Seg_img)),2)
# data=my_pca(np.concatenate((image,Seg_img),2))
data=image #0.62
# data=Seg_img  #0.83
# data=my_pca(image )
# data=my_pca(Seg_img)
# data=preprocess(data)
print(data.shape)
# data=np.ceil((data-data.min())/(data.max()-data.min()))

data=data.reshape([data.shape[0]*data.shape[1],data.shape[2]])
labels=labels.reshape([labels.shape[0]*labels.shape[1],1])
remain_index=np.argwhere(labels!=0)[:,0]
def print_metrics(y_true,y_pred):
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
    print('--------------------')
    print('overall_accuracy: {0:f}'.format(overall_accuracy))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:',kp)


trainX,testX,trainY,testY=train_test_split(data[remain_index],labels[remain_index],test_size=test_rate,random_state=2020)
# trainX,testX,trainY,testY=train_test_split(data[remain_index],labels[remain_index],train_size=train_size,stratify=labels[remain_index],random_state=2020)
print('训练数据：',trainX.shape,' 总可用样本：',remain_index.shape)
testY=testY.ravel()
trainY=trainY.ravel()
clf=SVC(decision_function_shape='ovr')
clf.fit(trainX,trainY)
y_pred=clf.predict(testX)
print_metrics(testY,y_pred)
print('len train',len(trainY),'  len test: ',len(testY),'  total: ',len(remain_index),' rate: ',len(trainY)/len(remain_index))


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
img=image.reshape([image.shape[0]*image.shape[1],image.shape[2]])

# clf2=SVC(decision_function_shape='ovr')
# clf2.fit(trainX,trainY)
# y_pred=clf2.predict(testX)
# metrics(testY,y_pred)

print('end')