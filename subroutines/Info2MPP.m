% JN Kather, extract microns per pixel MPP from Leica SVS
function currMPP = Info2MPP(currImInfo,channel)

    if isfield(currImInfo,'ImageDescription')
    currDescription = currImInfo(channel).ImageDescription;
    disp(currDescription);
    try
        st = strfind(currDescription,'MPP = ');         % find start
        currDescription = currDescription((st+6):end);  % strip before start
        en = strfind(currDescription,'|');      % find end
        if ~isempty(en)
        currDescription = currDescription(1:(en(1)-1));    % strip after end
        end
        currMPP = str2double(currDescription);  % get MPP (microns per pixel)
    catch
        warning('FAILED to parse description');
        currMPP = [];
    end
    else
        warning('no ImageDescription field');
        currMPP=[];
    end
   
end