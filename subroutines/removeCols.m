function tableIn = removeCols(tableIn,ColNames)

for i = 1:numel(ColNames)
    try
        tableIn.(ColNames{i}) = [];
    catch
        warning(['could not remove column ',ColNames{i}]);
    end
end

end