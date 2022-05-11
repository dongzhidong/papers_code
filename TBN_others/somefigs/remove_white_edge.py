import numpy as np 
import os 
import re 
import cv2
file_list=[x  for x in os.listdir() if re.match("HybridSN",x[:8])]
file_list=[x  for x in file_list if re.match("single.png",x[-10:])]
print("prepare to process...")
print(file_list)
def cut_row(img):
    while 1:
        im=img[0,:,1]
        im255=np.ones_like(im)*255
        if (im!=im255).any():
            break
        img=img[1:,:,:]
    # cv2.imshow("fn_cut",img)
    # cv2.waitKey()
    while 1:
        im=img[-1,:,1]
        im255=np.ones_like(im)*255
        if (im!=im255).any():
            break
        img=img[:-1,:,:]
    return img


def cut_col(img):
    while 1:
        im=img[:,0,1]
        im255=np.ones_like(im)*255
        if (im!=im255).any():
            break
        img=img[:,1:,:]
    while 1:
        im=img[:,-1,1]
        im255=np.ones_like(im)*255
        if (im!=im255).any():
            break
        img=img[:,:-1,:]
    return img

for fn in file_list:
    f=cv2.imread(fn)
    img=cut_row(f)
    img=cut_col(img)
    save_name=fn[:-4]+"_cut.png"
    cv2.imwrite(save_name,img)
    print("success ",save_name )
    # cv2.imshow("fn_cut",img)
    # cv2.waitKey()
    # print("pause")

print("end")


