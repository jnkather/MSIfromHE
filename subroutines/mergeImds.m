% JN Kather 2019, merge image datastores

function imdsResult = mergeImds(varargin)

    allNames = []; allLabels = [];

    for i = 1:nargin
        allNames = [allNames;varargin{i}.Files];
        allLabels = [allLabels;varargin{i}.Labels];
    end
    
    imdsResult = imageDatastore(allNames,'Labels',allLabels);
end