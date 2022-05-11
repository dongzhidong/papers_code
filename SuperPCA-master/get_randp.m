

clc;clear;
addpath(genpath(cd));
database='Houston2013';
% load Indian_pines_randp.mat;
% iprandp=randp;
%% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    num_Pixel        =   100;
elseif strcmp(database,'Salinas')
    load Salinas_corrected;load Salinas_gt;
    data3D = salinas_corrected;        label_gt = salinas_gt;
    num_Pixel        =   100;
elseif strcmp(database,'PaviaU')    
    load PaviaU;load PaviaU_gt;
    data3D = paviaU;        label_gt = paviaU_gt;
    num_Pixel        =   20;
 elseif strcmp(database,'Houston2013')    
    load Houston2013.mat;load Houston2013_gt.mat;
    data3D = Houston;        label_gt = labels;
    num_Pixel        =   20;
end
iter_num=10;
randp={};
[w,h]=size(label_gt);
label_gt=reshape(label_gt,w*h,1);
for iter =1:iter_num
    classnum=unique(label_gt);
    for i=1:length(classnum)-1
       ci=classnum(i+1);
       K=length(find(label_gt==i));
       ind=randperm(K);
       randpp{1,i}=ind;
    end 
    randp{1,iter}=randpp; 
end
save_name=strcat(['.\datasets\',database,'_randp.mat']);
save(save_name,'randp');


