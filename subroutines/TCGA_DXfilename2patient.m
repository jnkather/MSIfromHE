function outName = TCGA_DXfilename2patient(currName)

% extract patient ID from TCGA format string
dots = strfind(currName,'.');
outName = currName(1:(dots(1)-1));

end