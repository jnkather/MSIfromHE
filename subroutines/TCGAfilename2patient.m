function [allUniqNames,allNames] = TCGAfilename2patient(allNames,dashPosition)

% dashPosition is 3 for patient-level and 6 for slide level

% extract patient ID from TCGA format string
for i=1:numel(allNames)
    currName = allNames{i};
    dashes = strfind(currName,'-');
    if numel(dashes)<dashPosition
       % add the last point (empirically tested 29 Jan 2019)
       points = strfind(currName,'.');
       dashes = [dashes,points(end)];
    end
    allNames{i} = currName(1:(dashes(dashPosition)-1));
end

allUniqNames = unique(allNames);

end