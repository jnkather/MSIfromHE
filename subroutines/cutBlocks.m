function decision = cutBlocks(structIn,cnst,currImName)
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
    if sum(sum(bwIm<=1))<(numel(bwIm))
    % save image
        imwrite(blockIn,[cnst.outDir,currImName,'-blk-',randseq(12,'alphabet','amino'),'.png']);
        decision = 4; % code for success
    else
        warning('moving block to failures');
        imwrite(blockIn,[cnst.failDir,currImName,'-blk-',randseq(12,'alphabet','amino'),'.png']);
        decision = 1; % code for fail
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