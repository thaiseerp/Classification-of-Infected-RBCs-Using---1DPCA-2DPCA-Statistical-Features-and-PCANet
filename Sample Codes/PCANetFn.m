% ADD META TAGS

clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
make;

% Loading Data 
% load('./Data/FullImdb'); 
% %
% Set = FullImdb.images.set';
% TrnData = FullImdb.images.data(:,:,Set(:,end)==1); 
% TrnLabels = FullImdb.images.label(:,Set(:,end)==1)';
% ValData = FullImdb.images.data(:,:,Set(:,end)==2);
% ValLabels = FullImdb.images.label(:,Set(:,end)==2)';

% TestData = FullImdb.images.data(:,:,Set(:,end)==3);
% TestLabels = FullImdb.images.label(:,Set(:,end)==3)';
% 
% clear FullImdb; clear Set;
load('data.mat'); load('label.mat');

TrnData = data;
TrnLabels = label;

% nValImg = length(ValLabels);
% nTestImg = length(TestLabels);


    PCANet.NumStages = 2;
    PCANet.PatchSize = [9 9];
    PCANet.NumFilters = [8 8];
    PCANet.HistBlockSize = [9 9]; 
    PCANet.BlkOverLapRatio = 0;
    PCANet.Pyramid = [];

%% PCANet Training

fprintf('\n ====== PCANet Training ======= \n')
TrnData_ImgCell = Img2Cell(TrnData); % convert columns in TrnData to cells 
clear TrnData; 
tic;
[ftrain6720,PCAFilter6720,BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
save('PCAFilter6720','PCAFilter6720');
toc;
%%

ftrain2 = ftrain2'*PCA20Comp;
%%
ftrainNo = ftrain2./repmat(max(ftrain2),size(ftrain2,1),1);
ftrainNor = (ftrain2-repmat(min(ftrain2),size(ftrain2,1),1))...
    ./repmat(max(ftrain2)-min(ftrain2),size(ftrain2,1),1);
%%
%clear TrnData_ImgCell; 
%

fprintf('\n **** Training SVM **** \n')
tic;
SVMModel6720Linear = svmtrain(ftrain6720', TrnLabels, 'method', 'SMO', 'kernel_function', 'linear');
save('SVMModel6720Linear','SVMModel6720Linear'); clear SVMModel6720Linear;
toc;

tic;
SVMModel6720rbf = svmtrain(ftrain6720', TrnLabels, 'method', 'SMO', 'kernel_function', 'rbf');
save('SVMModel6720rbf','SVMModel6720rbf'); clear SVMModel6720rbf;
toc;

%clear ftrain; 

%% PCANet Feature Extraction and Validation

ValData_ImgCell = Img2Cell(ValData); % convert columns in TestData to cells 
% clear ValData; 

fprintf('\n ====== PCANet Validating ======= \n')

tic;
PrValLbl = zeros(nValImg,1);
for idx = 1:1:nValImg
    if 0==mod(idx,200); display(['Classifying ' num2str(idx) 'th Validation sample...']); end
    fval = PCANet_FeaExt(ValData_ImgCell(idx),PCAFilter,PCANet); % extract a test feature using trained PCANet model 
    PrValLbl(idx,1)       = svmclassify(SVMModel, fval');  
end
    ConfnMatVal  = confusionmat(ValLabels,PrValLbl);

%% PCANet Feature Extraction and Testing 

TestData_ImgCell = Img2Cell(TestData); % convert columns in TestData to cells 
 clear TestData; 

fprintf('\n ====== PCANet Testing ======= \n')

tic;
PrTsLbl = zeros(nTestImg,1);
for idx = 1:1:nTestImg
    if 0==mod(idx,1000); display(['Classifying ' num2str(idx) 'th Test sample...']); end
    fval = PCANet_FeaExt(TestData_ImgCell(idx),PCAFilter,PCANet); % extract a test feature using trained PCANet model 
    fval = fval'*PCA20Comp;
    PrTsLbl(idx,1)       = svmclassify(SVMModel, fval);  
end
    ConfnMatTest  = confusionmat(TestLabels,PrTsLbl);


    