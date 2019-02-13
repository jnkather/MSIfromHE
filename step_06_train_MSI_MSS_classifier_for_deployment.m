% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 06: 
% train a convolutional neural network model to classify MSI vs non-MSI
% from histological images. This trained neural network model can later be
% applied to pre-tesselated images (tiles in a folder) or whole slide
% images (deployment on the fly)

clear variables, close all, clc

warning('on','nnet_cnn:warning:GPULowOnMemory') % set low memory warning on
addpath(genpath('./subroutines/'));
mkdir('./dump');

% specify data sources
image_inputPath = 'E:\DX_TILES_NORM_SPLIT\TRAIN\';   % training set
image_externalTesting = 'E:\DX_TILES_NORM_SPLIT\TEST_noBalance\';  % path to
% testing set which has been split from training set on the level of
% patients. 

loadPreviousProgress = false; % continue where you stopped before, default false
currFn = 'classiMSSvsMSIMUT_CRC_DX'; % current filename to identify the network model later on

% specify learning parameters 
hyperparam.InitialLearnRate = 1e-5; % initial learning rate, will be overwritten later on
%hyperparam.Momentum = 0.9;          % momentum for stochastic gradient descent
hyperparam.ValidationFrequency = 256; % check validation performance every N iterations, 500 is 3x per epoch
hyperparam.ValidationPatience = 3; % wait N times before abort
hyperparam.L2Regularization = 1e-4; % optimization L2 constraint
hyperparam.MiniBatchSize = 256;    % mini batch size, limited by GPU RAM, default 256 on P6000
hyperparam.MaxEpochs = 100;           % max. epochs for training, default 100
hyperparam.hotLayers = 0;        % how many layers from the end are not frozen, will be overwritten
hyperparam.learnRateFactor = 2; % learning rate factor for rewired layers
hyperparam.ExecutionEnvironment = 'gpu'; % environment for training and classification
hyperparam.PixelRangeShear = 5;  % max. shear (in pixels) for image augmenter
allHyperparam = fieldnames(hyperparam);

%% READ ALL IMAGES
allImages = imageDatastore(image_inputPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% READ TRAINING IMAGES
% split the training images into a true training set of 85%, a validation
% set of 12.5% to avoid overfitting during the training stage and a test
% set of 2.5% which is currently not being used. This split is happening on
% the level of tiles within the training set
[training_set, validation_set, testing_set] = splitEachLabel(allImages,.85,.125,.025);
training_tbl = countEachLabel(training_set) %#ok
training_categories = training_tbl.Label; % extract category labels (from folder name)
figure, imshow(preview(training_set)); % show preview image
disp('successfully LOADED TRAINING images');

%% DATA AUGMENTATION FOR TRAINING --- REFLECTION  AND TRANSLATION (be careful 12/12/2018)
imageAugmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true,...
    'RandXTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear],...
    'RandYTranslation',[-hyperparam.PixelRangeShear,hyperparam.PixelRangeShear]);
disp('successfully LOADED image augmenter');

%% READ TESTING IMAGES
%testing_set = imageDatastore(testing_inputPath,'IncludeSubfolders',true,'LabelSource','foldernames');
testing_tbl = countEachLabel(testing_set) %#ok
testing_categories = testing_tbl.Label; % extract category labels (from folder name)
disp('successfully LOADED INTERNAL TESTING images');
figure, imshow(preview(testing_set)); % show preview image

%% READ EXTERNAL TESTING IMAGES
exttest_set = imageDatastore(image_externalTesting,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
disp('successfully LOADED EXTERNAL TESTING images');
figure, imshow(preview(exttest_set)); % show preview image

%% TRAIN NETWORK
experimentCounter = 0;
skipCounter = 0;
allModels = ({'resnet18'}); % use the resnet18 model (residual learning)

if loadPreviousProgress
    load(['./dump/',currFn,'.mat']);
    skipCounter = numel(results.networkName);
end

for InitialLearnRate = [1e-5] % initial learn rate, default 1e-5
hyperparam.InitialLearnRate = InitialLearnRate;
for hotLayers = [10] % these are layers with learnable properties, default 10
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
PerItemAccuracy = mean(predLabels == testing_set.Labels);

resized_ext_testing_set = augmentedImageDatastore(imageInputSize,exttest_set);
[predLabelsExternal,predScoresExternal] = classify(myNet, resized_ext_testing_set, ...
    'ExecutionEnvironment',hyperparam.ExecutionEnvironment);
PerItemAccuracyExternal = mean(predLabelsExternal == exttest_set.Labels);

% assess accuracy, show confusion matrix
%plotMyConfusion(testing_set.Labels,predLabels,cellstr(unique(testing_set.Labels))); % plot confusion matrix
plotMyConfusion(exttest_set.Labels,predLabelsExternal,cellstr(unique(exttest_set.Labels))); % plot confusion matrix

title(['classification with ',nnmodel,' -> overall per image accuracy in internal test set is ',...
    num2str(round(100*PerItemAccuracy)),'%',newline,'overall accuracy in external test set is ',...
    num2str(round(100*PerItemAccuracyExternal)),'%']);

% save results
results.networkName{experimentCounter} = nnmodel;
for j=1:size(allHyperparam)
results.(allHyperparam{j}){experimentCounter} = hyperparam.(allHyperparam{j});
end
results.trainTime{experimentCounter} = trainTime;
results.PerItemAccuracy{experimentCounter} = PerItemAccuracy;
results.PerItemAccuracyExternal{experimentCounter} = PerItemAccuracyExternal;

% save model
save(['./dump/',currFn,'.mat'],'results');
save(['./dump/',currFn,'_lastnet.mat'],'myNet');

drawnow
end
end
end

struct2table(results') %#ok

% afterwards, make sure to delete random images from the training folder
% until MSI and MSS contain the same number of images. The file names are
% preceded by a random string, so picking a random subset is easy.
