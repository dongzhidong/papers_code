clear;close all;clc;
%% load data
% name='houston';
name='Houston2013';
if strcmp(name,'houston')
    image=imread('houston.tif');
%     nC=50;
elseif strcmp(name,'Houston2013')
    load Houston2013.mat
    image=Houston;
end
image=double(image);
[w,h,b]=size(image);

nC_list=[400,500,600];
for nCi=1:length(nC_list)
    nC=nC_list(nCi);
   
%     nC=400;
    
    %% paramters
    lambda_prime = 0.2;sigma = 5.0; 
    conn8 = 1; 
    results=zeros(w,h,b);
    %% 
    tic
    disp(['分割尺度：',num2str(nC),'...']);
    for i=1:b
        disp(['处理通道',num2str(i),'...']);
        pause(0.0005)
        img=image(:,:,i);
        img=reshape(mapminmax(reshape(img,1,w*h),0,255),w,h);
    %     img=reshape(reshape(img,1,w*h),w,h);
        [labels] = mex_ers(double(img),nC,lambda_prime,sigma);
        results(:,:,i)=labels;
    end
    toc
    savepath='E:\pycharmworkspace\HSI\ERSMatlab\output\';
    save_name=[savepath,name,'_255seg',num2str(nC),'.mat'];
    save(save_name,'results')
end

