function decision = cutBlocksNeuralCascade(structIn,cnst,currImName,myNet_level1,myNet_level2,ref_image)
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
        [predLabel,predScore] = classify(myNet_level1, resizedIm, ...
            'ExecutionEnvironment',cnst.ExecutionEnvironment);    
        [mm,mi] = max(predScore);
        disp(['tile predicted as ',char(predLabel), ' with confidence ',num2str(round(mm,2))]);

        
        if strcmp(char(predLabel),'TUMSTU') % this is a tumor tile
            disp('> starting level 2 classification');
         	% do the level 2 classification
              resizedIm = Norm(resizedIm, ref_image, 'Macenko', 255, 0.15, 1, 0);
            disp('> > normalization done');
             [predLabel2,predScore2] = classify(myNet_level2, resizedIm, ...
              'ExecutionEnvironment',cnst.ExecutionEnvironment);  
              [mm2,mi2] = max(predScore2);
              disp(['> > > level 2 predicted as ',char(predLabel2), ' with confidence ',num2str(round(mm2,2))]);
              decision = 100*predScore2(1);%
        else % this is a non-tumor tile
           % mi2 = 0; 
            disp('> non-tumor tile');
            decision = -mi;%
        end
        % 4+mi + 100*mi2; % code for success + tissue class
    else % this is a black (failed) image
        warning('> failed tile!');
        decision = -100; % code for fail 
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