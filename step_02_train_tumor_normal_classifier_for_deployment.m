% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 02:
% train a resnet18 model to discriminate tumor vs. normal (normal being
% dense and loose normal tissue as described in the data set). The final
% classifier is saved for subsequent deployment

clear variables, close all, clc

warning('on','nnet_cnn:warning:GPULowOnMemory') % set low memory warning on
addpath(genpath('./subroutines/'));
mkdir('./dump');

% specify data sources
image_inputPath = 'D:\NCT_512_3CL_dataset\';  % parent folder for the data set
% which needs to be downloaded from https://zenodo.org/record/2530789. Each
% class is one subfolder

loadPreviousProgress = false; % continue where you stopped before
currFn = 'classi3fin'; %curr filename

% specify learning parameters 
hyperparam.InitialLearnRate = 1e-5; % initial learning rate
hyperparam.ValidationFrequency = 150; % check validation performance every N iterations, 500 is 3x per epoch
hyperparam.ValidationPatience = 10; % wait N times before abort
hyperparam.L2Regularization = 1e-4; % optimization L2 constraint
hyperparam.MiniBatchSize = 64;      % mini batch size, limited by GPU RAM, default 100 on Titan, 500 on P6000
hyperparam.MaxEpochs = 150;           % max. epochs for training, default 15
hyperparam.hotLayers = 100;        % how many layers from the end are not frozen
hyperparam.learnRateFactor = 2; % learning rate factor for rewired layers
hyperparam.ExecutionEnvironment = 'gpu'; % environment for training and classification
hyperparam.PixelRangeShear = 5;  % max. shear (in pixels) for image augmenter
allHyperparam = fieldnames(hyperparam);

%% READ ALL IMAGES
allImages = imageDatastore(image_inputPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% READ TRAINING IMAGES
[training_set, validation_set, testing_set] = splitEachLabel(allImages,.7,.15,.15);
training_tbl = countEachLabel(training_set) %#ok
training_categories = training_tbl.Label; % extract category labels (from folder name)
figure, imshow(preview(training_set)); % show preview image
disp('successfully LOADED TRAINING images');

%% DATA AUGMENTATION FOR TRAINING --- REFLECTION AND TRANSLATION 
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandXTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear],...
    'RandYTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear]);
disp('successfully LOADED image augmenter');

%% READ TESTING IMAGES
%testing_set = imageDatastore(testing_inputPath,'IncludeSubfolders',true,'LabelSource','foldernames');
testing_tbl = countEachLabel(testing_set) %#ok
testing_categories = testing_tbl.Label; % extract category labels (from folder name)
disp('successfully LOADED TESTING images');
figure, imshow(preview(testing_set)); % show preview image

%% TRAIN NETWORK
experimentCounter = 0;
skipCounter = 0;
allModels = {'resnet18'}; 

if loadPreviousProgress
    load(['./dump/',currFn,'.mat']);
    skipCounter = numel(results.networkName);
end

for InitialLearnRate = [5e-6]
hyperparam.InitialLearnRate = InitialLearnRate;
for hotLayers = [20]
hyperparam.hotLayers = hotLayers;
for i = 1:numel(allModels)
experimentCounter = experimentCounter+1;
nnmodel = allModels{i};
disp(['starting to work with ',nnmodel]);

% skip N times
if skipCounter>0
    skipCounter = skipCounter-1;
    disp(['skipping because skipCounter = ',num2str(skipCounter)]);
    continue;
end

% get network
[lgraph,imageInputSize,networkType] = getAndModifyNet(nnmodel,hyperparam,numel(unique(training_set.Labels)));
augmented_training_set = augmentedImageSource(imageInputSize,training_set,...
    'DataAugmentation',imageAugmenter);
disp(['sucessfully loaded&modified network, input size is ', num2str(imageInputSize)]);

%% TRAIN
% use image augmenter for resizing validation set images
resized_validation_set = augmentedImageDatastore(imageInputSize,validation_set);
opts = getTrainingOptions(hyperparam,resized_validation_set);
t = tic;
myNet = trainNetwork(augmented_training_set, lgraph, opts);
trainTime = toc(t);

%% DEPLOY
% use image augmenter for resizing testing set images
resized_testing_set = augmentedImageDatastore(imageInputSize,testing_set);
[predLabels,predScores] = classify(myNet, resized_testing_set, ...
    'ExecutionEnvironment',hyperparam.ExecutionEnvironment);

% assess accuracy, show confusion matrix
PerItemAccuracy = mean(predLabels == testing_set.Labels);
allgroups = cellstr(unique(testing_set.Labels));
plotMyConfusion(testing_set.Labels,predLabels,allgroups); % plot confusion matrix
title(['classification with ',nnmodel,' -> overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%']);

% save results
results.networkName{experimentCounter} = nnmodel;
for j=1:size(allHyperparam)
results.(allHyperparam{j}){experimentCounter} = hyperparam.(allHyperparam{j});
end
results.trainTime{experimentCounter} = trainTime;
results.PerItemAccuracy{experimentCounter} = PerItemAccuracy;

% save model
save(['./dump/',currFn,'.mat'],'results');
save(['./dump/',currFn,'_lastnet.mat'],'myNet');
end
end
end

struct2table(results) %#ok