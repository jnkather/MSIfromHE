function decision = cutBlocksNeural(structIn,cnst,currImName,myNet,ref_image)
    blockIn = structIn.data;

    currSize = size(blockIn);
    if (currSize(1) == currSize(2)) % discard non-square blocks 

    % keep only non-background blocks, according to Coudray et al., Nat Med 2018
    % criterion
    bwIm = mean(blockIn,3); % black and white image
    if sum(sum(bwIm>=220))<(0.5*numel(bwIm))
             
    % resize block to 512x512
    blockIn = imresize(blockIn,[cnst.finalBlockSize,cnst.finalBlockSize]);
    
    if cnst.verbose
    imshow(blockIn);
    drawnow
    end
    
    % clean up file name
    currDot = strfind(currImName,'.');
    currImName = currImName(1:(currDot(1)-1));
    
    % identify failed (black) images
    if sum(sum(bwIm<=1))<(numel(bwIm)) % THIS IS A NON-BACKGROUND, NON-FAILED, NON-EDGE image       
        % classify with neural network
        resizedIm = imresize(blockIn,[cnst.NeuralInputSize,cnst.NeuralInputSize]);
        [predLabel,predScore] = classify(myNet, resizedIm, ...
            'ExecutionEnvironment',cnst.ExecutionEnvironment);    
        [mm,mi] = max(predScore);
        disp(['tile predicted as ',char(predLabel), ' with confidence ',num2str(round(mm,2))]);
        % normalize tile (after classification, because classifier works
        % well in native image).
        if cnst.NormalizeOnTheFly
        blockIn = Norm(blockIn, ref_image, 'Macenko', 255, 0.15, 1, cnst.verbose);
        disp('normalization done');
        end
        
        if strcmp(char(predLabel),'TUMSTU') % this is a tumor tile
            imwrite(blockIn,[cnst.outDir,'blk-',randseq(12,'alphabet','amino'),'-',currImName,'.png']);
        else % this is a non-tumor tile
            imwrite(blockIn,[cnst.nonTuDir,'blk-',randseq(12,'alphabet','amino'),'-',currImName,'.png']);
        end
        decision = 4+mi; % code for success + tissue class
    else % this is a black (failed) image
        warning('moving block to failures');
        imwrite(blockIn,[cnst.failDir,currImName,'-blk-',randseq(12,'alphabet','amino'),'.png']);
        decision = 1; % code for fail 
        error('failed');
    end
    else
        if cnst.verbose
        disp('discarding background block');
        end
        decision = 2; % code for background
    end
    else
        if cnst.verbose
        disp('discarding edge');
        end
        decision = 3; % code for edge
    end
    
end