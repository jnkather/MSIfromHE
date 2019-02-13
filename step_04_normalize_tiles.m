% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 04:
% for each tumor image tile, load it and color-normalize it using the
% reference image

clear variables, close all, clc
addpath(genpath('./subroutines_normalization/'));
warning ('on','all');
ref_image = imread('Ref.png'); % reference image for image color normalization
sq = @(varargin) varargin';

cnst.verbose = false; % show intermediate steps on screen
cnst.tilesDir = 'D:\Colorectal_MSI_Project\CRC_DX_TILES\'; % load the non-normalized tiles from here
cnst.targetDir = 'D:\Colorectal_MSI_Project\CRC_DX_TILES_NORM\'; % save the normalized tiles here
cnst.failTilesDir = 'D:\Colorectal_MSI_Project\DX_TILES_NORM_FAIL\'; % save failed tiles here 
% the failTilesDir should be empty because no failures should happen.
% However, if it happens (e.g. because the process was manually aborted),
% the incomplete tiles will be saved here

cnst.ImOutputSize = 224;
mkdir(cnst.targetDir);
mkdir(cnst.failTilesDir);

% read all source files 
allSourceFiles = dir([cnst.tilesDir,'*.png']);
allSourceFileNames = sq(allSourceFiles(:).name);
% read all target files
allTargetFiles = dir([cnst.targetDir,'*.png']);
allTargetFileNames = sq(allTargetFiles(:).name);

[C,ia,ib] = intersect(allSourceFileNames,allTargetFileNames);
disp(['there are ',num2str(numel(allSourceFileNames)),' source tiles']);
allSourceFileNames(ia) = []; % remove processed tiles
disp(['there are ',num2str(numel(allSourceFileNames)),' tiles left to process']);


for i=1:numel(allSourceFileNames)
       currFn = allSourceFileNames{i};
    try % run the color normalization
        currIm = imread([cnst.tilesDir,currFn]);
        currIm = Norm(currIm, ref_image, 'Macenko', 255, 0.15, 1, cnst.verbose);
        currIm = imresize(currIm,[cnst.ImOutputSize,cnst.ImOutputSize]);
        imwrite(currIm,[cnst.targetDir,currFn]);
        trial = 10000;
    catch % if there was an error, save the file in failed dir
        warning('fail')
        movefile([cnst.tilesDir,currFn],[cnst.failTilesDir,currFn]);
        trial = 10000;
    end

    if mod(i,500)==1 % print the status from time to time
        disp(['finished file ',num2str(i),' of ',num2str(numel(allSourceFileNames))]);
    end
end