% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 05:
% this is an important step: we will split all image tiles into train and
% test set. This is done on a patient level, that means each patient will
% be assigned to the train or test group and then all tiles of this patient
% will be moved to the train or test folder

clear variables, close all, clc
addpath(genpath('./subroutines/'));
sq = @(varargin) varargin';
doCopy = true; % if false, the images will not be copied (simulation run)

% define the path to the normalized tiles
cnst.tileInputPath = 'D:\Colorectal_MSI_Project\DX_TILES_NORM\'; % source folder
cnst.tileOutputPath = 'E:\DX_TILES_NORM_SPLIT\'; % where to save the train and test set
mkdir(cnst.tileOutputPath);

% define the path to the clinical data table and to the hypermutation data table
% this requires Microsoft Excel to be installed on the machine
cnst.cliniDataTable = './cliniData/merged_TCGA_TUM_clini_table_v1.xlsx';

% read all tiles, might take a while
tic, 
allTiles = dir([cnst.tileInputPath,'*.png']); 
toc
tic, allTileNames = sq(allTiles.name); toc
disp('successfully read all tile names');

% read clinical data (merged data)
cliniDat = readtable(cnst.cliniDataTable);
cliniPatient = cliniDat.submitter_id;
cliniTumor = cliniDat.project_id;
cliniMSI = cliniDat.MSIStatus;
cliniHiMut = cliniDat.Tota_Mutation_cnt; % extract hypermutated samples. 
% hypermutated tumors will have a number here, all other will have NaN

%% MATCH IMAGE PATCHES WITH PATIENT LEVEL DATA FOR TABLE 1
matches = zeros(size(cliniPatient));
tic
for i = 1:numel(cliniPatient)
    currPat = cliniPatient{i}; % current patient name
    TF = contains(allTileNames,currPat); % match all tiles
    matches(i) = sum(TF); % how many tiles were matched to this patient
    if isempty(cliniMSI{i}) % if no MSI info is present, look for mutation data 
        % the reason for this is that we have few MSI patients and we do
        % not want to miss any because the MSI field is empty. This will
        % affect only very few patients (8 STAD and 6 COAD patients)
        if isnan(cliniHiMut(i)) % set status not hypermutated
            msistat{i} = 'NA'; % no MSI data, not hypermutated
        else % set status hypermutated
            msistat{i} = 'MUT'; % no MSI data, but hypermutated (highly probable MSI)
        end
    else % set status MSS or MSI
        msistat{i} = cliniMSI{i}; 
    end
    if mod(i,25)==1
    disp(['progress: ',num2str(round(i/numel(cliniPatient)*100,1))]);
    end
end
toc

% how many MSI and MSS patients could be matched
imgmatch = matches>0;
msimatch = strcmp(msistat,'MSI-H')'&imgmatch; % these are known MSI-H patients
mutmatch = strcmp(msistat,'MUT')'&imgmatch;   % these patients have no MSI status but are hypermutated
mssmatch = strcmp(msistat,'MSS')'&imgmatch;   % these patients are MSS

disp(['I matched ',num2str(sum(msimatch)),' MSI-H patients']);
disp(['I matched ',num2str(sum(mutmatch)),' non-MSI-H MUT patients']);
disp(['I matched ',num2str(sum(mssmatch)),' MSS patients']);

%% RANDOMIZE PATIENTS
rng(1); % reset the random number generator for reproducibility
figure(1),clf, hold on
traintest = 0.70; % split the data set in a 70/30 ratio (approximate)

msitrain = msimatch & rand(size(msimatch))<traintest;
muttrain = mutmatch & rand(size(mutmatch))<traintest;
msstrain = mssmatch & rand(size(mssmatch))<traintest;

plot(1:numel(imgmatch),msitrain,'r.'),plot(1:numel(imgmatch),muttrain,'b.'),plot(1:numel(imgmatch),msstrain,'g.')

disp(['Training cohort contains ',num2str(sum(msitrain)),' MSI-H patients']);
disp(['Training cohort contains ',num2str(sum(muttrain)),' non-MSI-H MUT patients']);
disp(['Training cohort contains ',num2str(sum(msstrain)),' MSS patients']);

% everybody not in training set goes to test set
msitest = msimatch & ~msitrain; 
muttest = mutmatch & ~muttrain;
msstest = mssmatch & ~msstrain;

plot(1:numel(imgmatch),msitest,'rx'),plot(1:numel(imgmatch),muttest,'bx'),plot(1:numel(imgmatch),msstest,'gx')

disp(['Test cohort contains ',num2str(sum(msitest)),' MSI-H patients']);
disp(['Test cohort contains ',num2str(sum(muttest)),' non-MSI-H MUT patients']);
disp(['Test cohort contains ',num2str(sum(msstest)),' MSS patients']);

%% COPY IMAGE TILES TO TRAIN AND TEST FOLDERS

% now copy the files to the actual folder. This might take a while.
if doCopy
moveImagesToFolder(cliniPatient,msitrain,cnst.tileInputPath,cnst.tileOutputPath,'msitrain');
moveImagesToFolder(cliniPatient,muttrain,cnst.tileInputPath,cnst.tileOutputPath,'muttrain');
moveImagesToFolder(cliniPatient,msstrain,cnst.tileInputPath,cnst.tileOutputPath,'msstrain');

moveImagesToFolder(cliniPatient,msitest,cnst.tileInputPath,cnst.tileOutputPath,'msitest');
moveImagesToFolder(cliniPatient,muttest,cnst.tileInputPath,cnst.tileOutputPath,'muttest');
moveImagesToFolder(cliniPatient,msstest,cnst.tileInputPath,cnst.tileOutputPath,'msstest');
end

