clear
clc
%%
file='IndianPines';
n_level=9;
switch file
    case 'Salinas'
        load Salinas_corrected.mat 
        size_img=[512,217,204];
        fea=reshape(salinas_corrected,[size(1)*size(2),size(3)]);
    case 'PaviaU'
        load Pavia_U.mat  
    case 'IndianPines'
        load indian_pines.mat
        size_img=[145,145,200];
end
fea = mapminmax(fea',0,1)';
[cof,sco]=pca(fea);
fea=sco(:,1);
fea = mapminmax(fea',0,1)';
fea=round(fea*n_level)/n_level;
im=reshape(fea,size_img(1),size_img(2))';
imshow(im,'Colormap',jet(255))


