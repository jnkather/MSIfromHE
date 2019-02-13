% JN Kather, NCT Heidelberg / RWTH Aachen, 2017-2018
% see separate LICENSE 
%
% This MATLAB script is associated with the following project
% "Deep learning can predict microsatellite instability directly 
% from histology in gastrointestinal cancer". Please consider citing this
% publication if you re-use the code
%
% Step 03:
% read all SVS whole slide images, process them tile-by-tile, find the
% tumor tiles and save them to another folder. This might take a while
% because >10^5 image tiles are created from >10^2 whole slide images. 

clear variables, close all, format compact, clc
addpath(genpath('./subroutines/')); % include our own subroutines

% the folder ./subroutines_normalization contains the color normalization
% toolbox which needs to be downloaded separately for license issues. Get
% it here: https://warwick.ac.uk/fac/sci/dcs/research/tia/software/sntoolbox/
addpath(genpath('./subroutines_normalization/'));

warning ('on','all'); % we want to see all runtime warnings

imageFolder = 'E:\TCGA-DX\'; % this is the path to the whole slide 
% SVS images which can be downloaded at https://portal.gdc.cancer.gov/

cnst.rescueFolder = 'E:\TCGA-DX-FIXED\'; % fallback folder for images that
% could not be loaded to to file format conversion issues - for example, this
% helps if the download was incomplete. 

cnst.tiffPage = 1;
cnst.useParallel = false; % does not work for SVS, only for TIFF
cnst.finalBlockSize = 512; 
cnst.verbose = false; % show intermediate steps on screen
cnst.outDir = 'D:\Colorectal_MSI_Project\CRC_DX_TILES\';    % store tumor tiles
cnst.failDir = 'D:\Colorectal_MSI_Project\CRC_DX_FAIL\';    % store failed tiles (should be empty after run)
cnst.nonTuDir = 'D:\Colorectal_MSI_Project\CRC_DX_NONTUMOR\'; % store non-tumor tiles
% make sure to use only colon and rectal cancer
cnst.projectOfInterest = {'TCGA-COAD','TCGA-READ'}; 
cnst.NormalizeOnTheFly = false; % false = normalize tiles afterwards, this is the default
mkdir(cnst.outDir);
mkdir(cnst.failDir);
mkdir(cnst.nonTuDir);
cnst.doCut = true; % do the operation, set false to simulate only
cnst.ExecutionEnvironment= 'gpu'; % environment for tu vs normal classification
cnst.NeuralInputSize = 224; % this is the input layer size, 224 for resnet18
sq = @(varargin) cell2mat(varargin);
sq2 = @(varargin) varargin';
ref_image = imread('Ref.png'); % reference image for image color normalization

% load tumor vs normal classifier network which was previously trained
myNet = load('./dump/resnet18_3cl_512_nonorm.mat');

% read all images in source folder
allImages = [dir([imageFolder,'*.svs']);dir([imageFolder,'*.tiff'])];
disp(['found ',num2str(numel(allImages)),' images in folder']);
allImageNames = sq2(allImages(:).name);
[allUniqNames,allPatientNames] = TCGAfilename2patient(allImageNames,3);

% read all images in target folders
tmpDir = dir([cnst.failDir,'*.png']);
allImNamesFail = sq2(tmpDir(:).name);

tmpDir = dir([cnst.outDir,'*.png']);
allImNamesTumor = sq2(tmpDir(:).name);

tmpDir = dir([cnst.nonTuDir,'*.png']);
allImNamesNonTumor = sq2(tmpDir(:).name);

% read metadata
currMetadata = readtable('./dump/allSlidesMetadata.csv','Delimiter',';');
allMetadataNames = currMetadata.submitter_id;
allMetadataProject = currMetadata.project_id;

% use only tumor type of interest
for i = 1:numel(allPatientNames)
    currMetadataRow = strcmp(allMetadataNames,allPatientNames{i});
    allProjects{i} = char(allMetadataProject(currMetadataRow)); 
end

% start to analyze images
failedNames = {''}; % preallocate
successfulImages =0; % count success
corruptFiles = 0;
for i=1:numel(allImages)  % fail 131
    disp([newline,'***',newline,'looking at image ',num2str(i),' of ',num2str(numel(allImages))]);
    currImName = allImages(i).name;
    disp(['current image name is ', char(currImName)]);
    currProject = allProjects{i};
    disp(['current project is ',currProject]);
    if any(contains(cnst.projectOfInterest,currProject))
        disp(['will resume because this is a project of interest']);
    else
        disp(['will skip because this is NOT a project of interest']);
        continue
    end
       
    % check if this image has already been processed
    currPatName = TCGA_DXfilename2patient(currImName);
    isPresentInFails = any(contains(allImNamesFail,currPatName));
    isPresentInTum = any(contains(allImNamesTumor,currPatName));
    isPresentInNontum = any(contains(allImNamesNonTumor,currPatName));
    if (isPresentInFails||isPresentInTum||isPresentInNontum)
        disp('will skip because this image has already been processed');
        if isPresentInFails
            corruptFiles = corruptFiles+1;
        end
        continue
    end
        
    currImPath = [imageFolder,currImName];
    try
    currImInfo = imfinfo(currImPath);
    catch
        warning(['could not read image info. Image is probably corrupt. Will skip']);
        corruptFiles = corruptFiles+1;
        pause(1)
        continue
    end
    % read image thumbnail
    %currThumb = imread(currImPath,5
    %imshow(currThumb));
    %drawnow
    
    % block process current image
    %[mm,mi] = max(sq(currImInfo.Width));
    [nn,ni] = sort(sq(currImInfo.Width));
    % show current thumbnail
    if numel(ni)>1
    figure
    subplot(1,3,1)
    imshow(imread(currImPath,ni(1)))
    subplot(1,3,2)
    imshow(imread(currImPath,ni(2))) 
    suptitle(strrep(currImName,'_',' '));
    drawnow
    else
        disp('could not show thumbnail because there is only one channel');
    end
    
    disp(['will use TIFF channel ' , num2str(ni(end)),' with Width = ',num2str(ni(end))]);
    
    currMPP = Info2MPPrescue(currImInfo,ni(end),... % extract MPP; if fail try to rescue
        cnst.rescueFolder,strrep(strrep(currImName,'.svs',''),'.tiff',''));     
    
    if ~isempty(currMPP) && ~isnan(currMPP)
    sqs = round(256/currMPP);  % square size is 512 px for 0.5 MPP. Blocks will be resized to 512
    disp(['current MPP is ',num2str(currMPP),', current window size is ', num2str(sqs)]);
    
    imData = PagedTiffAdapter(currImPath,ni(end)); 
    rng('shuffle');
    if cnst.doCut
        try
            disp('starting block processing');
            tic
            myFun = @(blk) cutBlocksNeural(blk,cnst,currImName,myNet.myNet,ref_image);
            outMask = blockproc(imData,[sqs,sqs],myFun,'UseParallel',cnst.useParallel);
            toc
            disp('finished block processing');
        catch
            warning('skipped image because of blockproc error');
            corruptFiles = corruptFiles+1;
            outMask = 1;
        end
        subplot(1,3,3)
        imagesc(outMask);
        axis equal tight off
        colorbar
        colormap parula
        caxis([0 8]);
        drawnow
        print(gcf,['./logs/',currImName,'_out_.png'],'-dpng','-r300');
        % check if there are failed tiles
        if any(outMask(:)==1)
            warning(['FAILURES in ',currImName]);
            failedNames = [failedNames,{currImName}];
        else
            disp('no failures');
        end
    else
        disp(['did not do anything (simulation run)']);
    end
    successfulImages = successfulImages+1;
    else % parse image metadata failed
        disp(['FAILED MPP detection in iteration ',num2str(i),'... will start next image']);
        corruptFiles = corruptFiles +1;
    end
    pause(0.01);
    close all
end

disp(['finished analyzing all images. Success: ',num2str(successfulImages)]);
disp(['ther were ',num2str(corruptFiles),' corrupt files']);