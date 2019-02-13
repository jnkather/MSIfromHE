% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 01:
% This script will train a classifier for tumor vs. normal tissue 
% and report validation accuracy of 5x cross validation. After running
% this, the second script (step 02) is used to train a tumor vs. normal
% classifier on the full data (no cross validation).

clear variables, close all, clc

% train a resnet18 model with learning rate 1e-5 and 20 hot layers to
% discriminate three classes of tissue. This will be used in the block
% processing function read and chop images to detect tumor on the fly
warning('on','nnet_cnn:warning:GPULowOnMemory') % set low memory warning on
addpath(genpath('./subroutines/'));
mkdir('./dump');

% specify data sources
image_inputPath = 'D:\NCT_512_3CL_dataset\';  % parent folder for the data set
% which needs to be downloaded from https://zenodo.org/record/2530789. Each
% class is one subfolder

loadPreviousProgress = false; % continue where you stopped before
currFn = 'classi3xval'; % current filename for saving log files

% specify learning parameters 
hyperparam.InitialLearnRate = 5e-6; % initial learning rate
hyperparam.L2Regularization = 1e-4; % optimization L2 constraint
hyperparam.MiniBatchSize = 64;      % mini batch size, limited by GPU RAM, default 100 on Titan, 500 on P6000
hyperparam.MaxEpochs = 5;          % max. epochs for training, default 15
hyperparam.hotLayers = 20;        % how many layers from the end are not frozen
hyperparam.learnRateFactor = 2; % learning rate factor for rewired layers
hyperparam.ExecutionEnvironment = 'gpu'; % environment for training and classification
hyperparam.PixelRangeShear = 5;  % max. xy translation (in pixels) for image augmenter
allHyperparam = fieldnames(hyperparam);

%% READ ALL IMAGES and prepare for cross validation
allImages = imageDatastore(image_inputPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[sub1,sub2,sub3,sub4,sub5] = splitEachLabel(allImages,.2,.2,.2,.2,.2);

%% DATA AUGMENTATION FOR TRAINING --- REFLECTION AND TRANSLATION 
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandXTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear],...
    'RandYTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear]);

%% START CROSS VALIDATED TRAINING
for cval = 1:5 % manual implementation of 5x cross validation
    disp(['*******',char(10),'STARTING CROSSVAL ',num2str(cval)]);
    switch cval
        case 1
            trainSet = sub1;  testSet = mergeImds(sub2,sub3,sub4,sub5);
        case 2                            
            trainSet = sub2;  testSet = mergeImds(sub1,sub3,sub4,sub5);
        case 3                           
            trainSet = sub3;  testSet = mergeImds(sub1,sub2,sub4,sub5);
        case 4                              
            trainSet = sub4;  testSet = mergeImds(sub1,sub2,sub3,sub5);
        case 5                       
            trainSet = sub5;  testSet = mergeImds(sub1,sub2,sub3,sub4);
    end 
 
% get network
[lgraph,imageInputSize,networkType] = getAndModifyNet('resnet18',hyperparam,numel(unique(trainSet.Labels)));
augmented_training_set = augmentedImageSource(imageInputSize,trainSet,...
    'DataAugmentation',imageAugmenter);
disp('sucessfully loaded&modified network');

%% TRAIN
opts = getTrainingOptions(hyperparam,[]);
t = tic;
myNet = trainNetwork(augmented_training_set, lgraph, opts);
trainTime = toc(t);

%% DEPLOY
% use image augmenter for resizing testing set images
resized_testing_set = augmentedImageDatastore(imageInputSize,testSet);
[predLabels,predScores] = classify(myNet, resized_testing_set, ...
    'ExecutionEnvironment',hyperparam.ExecutionEnvironment);

% calculate AUC
[ulabels,uia,uic]  = unique(testSet.Labels);
currClassIndx = 3; % tumstu
currClass = ulabels(currClassIndx)
currNeg = ulabels;
currNeg(currClassIndx) = [];
[X,Y,T,AUC,OR,SUBY,SUBYN] = perfcurve(uic,predScores(:,currClassIndx),currClassIndx);
AllAUC(cval) = AUC;
plot(X,Y,'k','LineWidth',1.8);
xlabel('FPR')
ylabel('TPR')
axis square
title([char(currClass),' AUC=',num2str(round(AUC,2))]);
drawnow
       

% assess accuracy, show confusion matrix
PerItemAccuracy(cval) = mean(predLabels == testSet.Labels);
allgroups = cellstr(unique(testSet.Labels));
plotMyConfusion(testSet.Labels,predLabels,allgroups); % plot confusion matrix
title(['-> overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%']);

drawnow

end

PerItemAccuracy
AllAUC
save('lastCrossval.mat','PerItemAccuracy');