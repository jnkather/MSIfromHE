% from https://blogs.mathworks.com/steve/2011/09/23/dealing-with-really-big-images-image-adapters/

classdef PagedTiffAdapter < ImageAdapter
    properties
        Filename
        Info
        Page
    end
    methods
        function obj = PagedTiffAdapter(filename, page)
            obj.Filename = filename;
            obj.Info = imfinfo(filename);
            obj.Page = page;
            obj.ImageSize = [obj.Info(page).Height obj.Info(page).Width];
        end
        function result = readRegion(obj, start, count)
            result = imread(obj.Filename,'Index',obj.Page,...
                'Info',obj.Info,'PixelRegion', ...
                {[start(1), start(1) + count(1) - 1], ...
                [start(2), start(2) + count(2) - 1]});
        end
        function result = close(obj) %#ok
        end
    end
end