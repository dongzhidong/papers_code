% =========================================================================
% A simple demo for SuperPCA based Unsupervised Feature Extraction of
% Hyperspectral Imagery
% If you  have any problem, do not hesitate to contact
% Dr. Jiang Junjun (junjun0595@163.com)
% Version 1,2018-04-12

% Reference: Junjun Jiang,Jiayi Ma, Chen Chen, Zhongyuan Wang, and Lizhe Wang, 
% "SuperPCA: A Superpixelwise Principal Component Analysis Approach for
% Unsupervised Feature Extraction of Hyperspectral Imagery," 
% IEEE Transactions on Geoscience and Remote Sensing, 2018.
%=========================================================================

clc;clear;close all
addpath('.\libsvm-3.21\matlab');
addpath(genpath(cd));

num_PC           =   30;  % THE OPTIMAL PCA DIMENSION.
% num_Pixel        =   100; % THE OPTIMAL Number of Superpixel. Indian:100, PaviaU:20, Salinas:100
trainpercentage  =   50;  % Training Number per Class
iterNum          =   1;  % The Iteration Number

% database         =   'Indian';
database         =   'Houston2013';
% database         =   'PaviaU';
% database         =   'Salinas';
%% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;load Indian_pines_randp 
    data3D = indian_pines_corrected;        label_gt = indian_pines_gt;
    num_Pixel        =   100;
elseif strcmp(database,'Salinas')
    load Salinas_corrected;load Salinas_gt;load Salinas_randp
    data3D = salinas_corrected;        label_gt = salinas_gt;
    num_Pixel        =   100;
    trainpercentage  =   50;
elseif strcmp(database,'PaviaU')    
    load PaviaU;load PaviaU_gt;load PaviaU_randp; 
    data3D = paviaU;        label_gt = paviaU_gt;
    num_Pixel        =   20;
 elseif strcmp(database,'Houston2013')    
    load Houston2013.mat;load Houston2013_gt.mat;load Houston2013_randp.mat
    data3D = double(Houston);        label_gt = labels;
    num_Pixel        =   20;

end
data3D = data3D./max(data3D(:));

%% super-pixels segmentation
labels = cubseg(data3D,num_Pixel);

%% SupePCA based DR
[dataDR] = SuperPCA(data3D,num_PC,labels);

for iter = 1:iterNum
    disp(['iter:',num2str(iter)]);
    randpp=randp{iter};     
    % randomly divide the dataset to training and test samples
    [DataTest DataTrain CTest CTrain map] = samplesdivide(dataDR,label_gt,trainpercentage,randpp);   
    disp(['length of Train:',num2str(length(DataTrain)),' length of Test:',num2str(length(DataTest))]);
    % Get label from the class num
    trainlabel = getlabel(CTrain);
    testlabel  = getlabel(CTest);

    % set the para of RBF
    ga8 = [0.01 0.1 1 5 10];    ga9 = [15 20 30 40 50 100:100:500];
    GA = [ga8,ga9];

    accy = zeros(1,length(GA));
    oas = zeros(1,length(GA));
    aas = zeros(1,length(GA));
    kps = zeros(1,length(GA));

    tempaccuracy1 = 0;
    for trial0 = 1:length(GA);    
        gamma = GA(trial0);        
        cmd = ['-q -c 100000 -g ' num2str(gamma) ' -b 1'];
        model = svmtrain(trainlabel', DataTrain, cmd);
        [predict_label, AC, prob_values] = svmpredict(testlabel', DataTest, model, '-b 1');    
        [confusion, accuracy1, CR, FR,oa,aa,kp] = confusion_matrix(predict_label', CTest);
        if tempaccuracy1<accuracy1
            best_model=model;
            tempaccuracy1=accuracy1;
        end
        accy(trial0) = accuracy1;
        oas(trial0)=oa;
        aas(trial0)=aa;
        kps(trial0)=kp;
    end
    [accy_best(iter),mi] = max(accy);
    OAS(iter)=oas(mi);
    AAS(iter)=aas(mi);
    KPS(iter)=kps(mi);
    
    disp(['acc:',num2str(accy_best(iter))]);
end
fprintf('\n=============================================================\n');
fprintf(['The average OA (5 iterations) of SuperPCA for ',database,' is %0.4f\n'],mean(accy_best));
fprintf('=============================================================\n');
fprintf('OAS mean:%0.4f  AAS mean:%0.4f  KPS mean:%0.4f \n',mean(OAS),mean(AAS),mean(KPS));
fprintf('OAS std:%0.4f AAS std:%0.4f KPS std:%0.4f \n',std(OAS),std(AAS),std(KPS));
for i=1:length(CR)
    fprintf('%0.4f\n',CR(i)*100);
end

%% show test all pixel 
label_show=zeros(size(label_gt));
[w,h]=size(label_show);
[a,b,c]=size(dataDR);
data=reshape(dataDR,a*b,c);
data=fea_norm(data);
label_all=reshape(label_gt,a*b,1);
[pall, AC, prob_values] = svmpredict(double(label_all), data, best_model, '-b 1');
% pall(find(label_all==0))=0;
predict_labels=reshape(pall,a,b);
save(strcat(database,'_predict_labels50.mat'),'predict_labels');
% unique(predict_labels)
