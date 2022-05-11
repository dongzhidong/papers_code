clear;close all;clc;
%% load data
% name='IndianPines';
% name='Salinas';
% name='PaviaU';
name='Houston2013';

if strcmp(name,'IndianPines')
    load Indian_pines_corrected.mat;
    image=indian_pines_corrected;
%     nC=50;
elseif strcmp(name,'PaviaU')
    load PaviaU.mat
    image=paviaU;
%     nC=50;
elseif strcmp(name,'Salinas')
    load Salinas_corrected.mat
    image=salinas_corrected;
%     nC=10;
elseif strcmp(name,'Houston2013')
    load Houston2013.mat
    image=Houston;
    image=double(image);
end
disp(['处理数据：',name,'...']);

[w,h,b]=size(image);
fea=reshape(image,[w*h,b]);

fea = mapminmax(fea',0,1)';
[cof,sco]=pca(fea);
fea=sco(:,1);
% fea = mapminmax(fea',0,255)';
img=reshape(fea,w,h);
% imshow(img/255)


nC_list=[100,200,300,400,500,600,700,800];
for nCi=1:length(nC_list)
    nC=nC_list(nCi);
    %% paramters
    lambda_prime = 0.2;sigma = 5.0; 
    conn8 = 1; 
    %% 
    tic
    disp(['分割尺度：',num2str(nC),'...']);
    
    img=reshape(mapminmax(reshape(img,1,w*h),0,255),w,h);
    %     img=reshape(reshape(img,1,w*h),w,h);
    [labels] = mex_ers(double(img),nC,lambda_prime,sigma);
    result=labels;
    %     show
    %     [bmap] = seg2bmap(labels,h,w);
    %     boundmap = img;
    %     boundmap(find(bmap>0))=125;
    %     [out] = random_color( double(img) ,labels,nC);
    %     figure(1)
    %     subplot(1,3,1);
    %     imshow(img,[]);
    %     title(['input image.',num2str(i)]);
    %     subplot(1,3,2);
    % %     imshow(boundmap,[]);
    %     imshow(boundmap,[]);
    %     title('superpixel boundary map');
    %     subplot(1,3,3);
    %     imshow(out,[]);
    %     title('randomly-colored superpixels');
    toc
    savepath='E:\pycharmworkspace\HSI\ERSMatlab\output\';
    save_name=[savepath,name,'_255segPCA',num2str(nC),'.mat'];
    save(save_name,'result')
end 
disp(['运行结束','...']);

