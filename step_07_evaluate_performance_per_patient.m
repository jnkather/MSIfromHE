% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 07: 
% here, we will use the trained neural network model to predict MSI for
% each tissue tile of patients in the test set. Please note that in this
% case, the images of the patients have already been tesselated and tumor
% tissue tiles have been saved in a folder and have been normalized. As an
% alternative to this script, the trained model can also predict tiles as
% they are extracted from a whole slide image (classification on the fly).

clear variables, close all, clc
addpath(genpath('./subroutines'));

NUID = 'd876c0e0'; % define the neural network ID to load the correct pre-trained model
tissueType = 'KR'; % specify DX or KR depending on whether you want to predict 
% diagnostic (FFPE, DX) or snap-frozen (kryo, KR) images
load(['./dump/classiMSSvsMSIMUT_',NUID,'_lastnet.mat']); % load trained network
try 
    % try to load the previous result if you just want to plot again
    load(['./dump/classiMSSvsMSIMUT_',NUID,'_09C.mat']);
    disp('loaded previous result');
catch % did not find a previously saved result file, go on with prediction
    disp('loading failed, will recompute');
myImds = imageDatastore(['D:\Colorectal_MSI_Project\',tissueType,'_TILES_NORM_SPLIT\TEST_noBalance'],...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imageInputSize = myNet.Layers(1).InputSize(1:2);
augMyImds = augmentedImageDatastore(imageInputSize,myImds);

disp('starting prediction');
tic
[YPred,predScoresExternal] =...
    classify(myNet,augMyImds,...
    'MiniBatchSize',1024,'ExecutionEnvironment','gpu');
toc
disp('done with prediction'); % 847 sec on P6000

tic
featuresExternal = activations(myNet,augMyImds,'fc');
toc
disp('done with feature extraction');

save(['./dump/classiMSSvsMSIMUT_',NUID,'_09C.mat'],...
    'YPred','myImds','predScoresExternal','featuresExternal'); % save dump
end

trueLabels = cellstr(myImds.Labels);
predLabels = cellstr(YPred);
predFeatures = squeeze(featuresExternal)';

% clean up file names
allFileNames = myImds.Files;
for i = 1:numel(allFileNames) % strip path
    ffirst = strfind(allFileNames{i},'-TCGA-');
    allFileNames{i} = allFileNames{i}((ffirst(1)+1):(end));
end

[allUniqNames,~] = TCGAfilename2patient(allFileNames,3); % for patient = 3, for slide = 5

% find classification accuracy on a per patient level
for i = 1:numel(allUniqNames)
    currName = allUniqNames{i};
    currMask = contains(myImds.Files,currName);
    
    % count hits
    trueMSIMUT = sum(strcmp(trueLabels(currMask),'MSIMUT'));
    predMSIMUT = sum(strcmp(predLabels(currMask),'MSIMUT'));
    
    trueMSS = sum(strcmp(trueLabels(currMask),'MSS'));
    predMSS = sum(strcmp(predLabels(currMask),'MSS'));
    
    trueIsMSIMUT(i) = trueMSIMUT / (trueMSIMUT+trueMSS);
    trueIsMSIMUTbool(i) = trueMSIMUT >= trueMSS;
    
    predIsMSIMUT(i) = predMSIMUT / (predMSIMUT+predMSS); % majority vote

    AllNumImages(i) = sum(currMask);
     
    disp(['patient ',currName,'; # tiles ',num2str(sum(currMask)),...
        '; true is MSIMUT: ',num2str(trueIsMSIMUT(i)),'; pred is MSIMUT: ',num2str(predIsMSIMUT(i)),...
        '; score: ',num2str(predIsMSIMUTScore(i))]);
end

predictor = predIsMSIMUT; 
removeMe = AllNumImages<10; % discard images with less than 10 tiles, this 
% will not affect the DACHS validation set because in this case, every
% patient has more than 100 tiles

% calculate stats as an initial sanity check. Do MSI image tiles lead to
% higher values of the MSI predictor than MSS image tiles? (spoiler: they do)
pred = predictor(~removeMe);
trueMSI = trueIsMSIMUTbool(~removeMe);
round(quantile(pred(trueMSI),[.5,.05,.95]),2)
round(quantile(pred(~trueMSI),[.5,.05,.95]),2)
[h, p, cid] = ttest2(pred(trueMSI),pred(~trueMSI))

% plot the ROC curve and calculate AUC
figure()
[X1,Y1,T1,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(trueIsMSIMUTbool(~removeMe)',...
    predictor(~removeMe)',1,'NBoot',500); 
% you can choose any cutoff and will get a different set of TPR/FPR as
% shown in the ROC curve. As an example cutoff, we can use the middle
% between true MSI and true MSS mean values. Again, the ROC curve carries
% the most comprehensive and unbiased information about the classifier
% performance and should be used to evaluate if the classifier is good
optcutoff = double(mean([mean(predictor(~trueIsMSIMUTbool & ~removeMe)),mean(predictor(trueIsMSIMUTbool & ~removeMe))]))
%plot(X,Y,'k','LineWidth',1.8);
hold on
plot(X1(:,1),Y1(:,1),'k','LineWidth',2) % mean
plot(X1(:,1),Y1(:,2),'Color',[.2 .2 .2],'LineWidth',.8) % lo CI
plot(X1(:,1),Y1(:,3),'Color',[.2 .2 .2],'LineWidth',.8) % hiCI
xlabel('FPR')
ylabel('TPR')
axis square
title(['MSIMUT per patient AUC = ',num2str(round(AUC(1),2)),... % char(10) is newline
    ' [',num2str(round(AUC(2),2)),', ',num2str(round(AUC(3),2)),']',char(10)]);
set(gcf,'Color','w');
drawnow
set(gca,'XTick',[0,1])
set(gca,'YTick',[0,1])
set(gca,'FontSize',18)
print(gcf,['./output_images/',NUID,'_ROC_ON_TEST_',tissueType,'.png'],'-dpng','-r900');

% this is optional: for any cutoff on the ROC curve, you can calculate
% stats such as PPV and NPV. Remember that the ROC AUC gives you a much
% more comprehensive view of the classifier performance which is why ROC
% AUC was used in similar studies (such as Coudray et al. Nat Med 2018)
figure()
cp = classperf(trueIsMSIMUTbool(~removeMe),predictor(~removeMe)>=optcutoff)
cmat = cp.DiagnosticTable
heatmap(cmat)
colorbar
caxis([0 max(cmat(:))]);
xlabel('true');
ylabel('predicted');

figure()
mygscatter(predIsMSIMUT(~removeMe),predictor(~removeMe),trueIsMSIMUTbool(~removeMe),100)

% produce output table
outputData.PatientID = allUniqNames(~removeMe);
outputData.nTiles = AllNumImages(~removeMe)';
outputData.predictedScore =  predictor(~removeMe)';
outputData.predictedMSI = double(predictor(~removeMe)>=optcutoff)';
outputData.trueMSI = double(trueIsMSIMUTbool(~removeMe))';
outputData.combinedMSI = strrep(strrep(strrep(strrep(...
    cellstr(num2str(outputData.predictedMSI + 2*outputData.trueMSI)),'0','tMSS,pMSS'),...
                         '1','tMSS,pMSI'),'2','tMSI,pMSS'),'3','tMSI,pMSI');
outputTable = struct2table(outputData)

% merge with clinical data table
cnst.cliniDataTable = './cliniData/merged_TCGA_TUM_clini_table_v1.xlsx';
cliniTable = readtable(cnst.cliniDataTable);

% save results
joinedTable = innerjoin(outputTable,cliniTable,'LeftKeys','PatientID','RightKeys','submitter_id');

% this analysis requires the MatSurv toolbox which cannot be included in
% this repository because of license issues. Please download it from the
% Mathworks file exchange.
myFilter = (joinedTable.cleanStage > 0) & (joinedTable.cleanStage <5);
MatSurv(joinedTable.OS_time(myFilter), joinedTable.OS(myFilter),...
    joinedTable.combinedMSI(myFilter),'GroupsToUse',{'tMSS,pMSI','tMSS,pMSS'})
print(gcf,['./output_images/',NUID,'_SURV2_TEST_',tissueType,'.png'],'-dpng','-r900');


