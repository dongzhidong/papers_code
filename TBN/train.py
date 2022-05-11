from paramters import *
from scipy import io as sio
import numpy as np
from tools import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
Seg_img=sio.loadmat(os.path.join(IndianPine_path,'Indian_seg.mat'))['seg_labels']
data=np.concatenate((my_pca(image),my_pca(Seg_img)),2)
data=preprocess(data)

name='SVM'

if name=='SVM':
    X,y=build_dataset(data)
    clf=SVC()
    clf.fit(X,y)
    clf.predict(test)

print('end')

