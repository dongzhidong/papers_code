from paramters import *
from scipy import io as sio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

image=sio.loadmat(os.path.join(IndianPine_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
image=np.reshape(image,[145*145,200])
pca=PCA(n_components=0.99)
pca.fit(image)
image1=pca.transform(image)
image1=np.reshape(image1,[145,145,-1])
plt.imshow(image1)
sio.savemat('pca.mat',{'image':image1})
print('end')
