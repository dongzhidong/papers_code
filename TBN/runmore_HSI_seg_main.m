clear;close all;clc;
%% load data
% name='IndianPines';
% name='Salinas';
name='PaviaU';
% name='houston';

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
end
[w,h,b]=size(image);

nC_list=[300,400,500,600];
for nCi=1:length(nC_list)
    nC=nC_list(nCi);
    %% paramters
    % nC=30;
    lambda_prime = 0.2;sigma = 5.0; 
    conn8 = 1; 
    results=zeros(w,h,b);
    %% 
    tic
    disp(['分割尺度：',num2str(nC),'...']);
    for i=1:b
        disp(['处理通道',num2str(i),'...']);
        pause(0.00005)
        img=image(:,:,i);
        img=reshape(mapminmax(reshape(img,1,w*h),0,255),w,h);
    %     img=reshape(reshape(img,1,w*h),w,h);
        [labels] = mex_ers(double(img),nC,lambda_prime,sigma);
        results(:,:,i)=labels;
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
    end
    toc
    savepath='E:\pycharmworkspace\HSI\ERSMatlab\output\';
    save_name=[savepath,name,'_255seg',num2str(nC),'.mat'];
    save(save_name,'results')
end 

