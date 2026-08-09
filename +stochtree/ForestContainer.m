classdef ForestContainer < handle
    %STOCHTREE.FORESTCONTAINER Stored posterior draws of a tree ensemble.
    %
    %   All sample indices exposed by this class are 1-based, matching MATLAB
    %   convention; they are translated to the C++ 0-based indices internally.
    %
    %   See also STOCHTREE.FOREST, STOCHTREE.BART.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = ForestContainer(numTrees, outputDimension, isLeafConstant, isExponentiated)
            if nargin < 2 || isempty(outputDimension), outputDimension = 1; end
            if nargin < 3 || isempty(isLeafConstant), isLeafConstant = true; end
            if nargin < 4 || isempty(isExponentiated), isExponentiated = false; end
            obj.Handle = mex_stochtree('container_create', double(numTrees), ...
                double(outputDimension), logical(isLeafConstant), logical(isExponentiated));
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function n = numSamples(obj)
            n = mex_stochtree('container_num_samples', obj.Handle);
        end

        function n = numTrees(obj)
            n = mex_stochtree('container_num_trees', obj.Handle);
        end

        function d = outputDimension(obj)
            d = mex_stochtree('container_output_dimension', obj.Handle);
        end

        function deleteSample(obj, sampleIndex)
            %DELETESAMPLE Drop a stored draw (1-based index).
            mex_stochtree('container_delete_sample', obj.Handle, double(sampleIndex) - 1);
        end

        function preds = predict(obj, dataset)
            %PREDICT Return an n-by-numSamples matrix of predictions.
            preds = mex_stochtree('container_predict', obj.Handle, dataset.Handle);
        end

        function raw = predictRaw(obj, dataset)
            %PREDICTRAW Return an n-by-outputDimension-by-numSamples array.
            raw = mex_stochtree('container_predict_raw', obj.Handle, dataset.Handle);
        end

        function raw = predictRawSingle(obj, dataset, sampleIndex)
            %PREDICTRAWSINGLE Raw leaf values for one draw (1-based index).
            raw = mex_stochtree('container_predict_raw_single', obj.Handle, ...
                dataset.Handle, double(sampleIndex) - 1);
        end

        function addToForest(obj, sampleIndex, value)
            mex_stochtree('container_add_to_forest', obj.Handle, ...
                double(sampleIndex) - 1, double(value));
        end

        function multiplyForest(obj, sampleIndex, multiplier)
            mex_stochtree('container_multiply_forest', obj.Handle, ...
                double(sampleIndex) - 1, double(multiplier));
        end

        function counts = splitCounts(obj, numFeatures)
            %SPLITCOUNTS Total splits on each feature, pooled over all draws.
            counts = mex_stochtree('container_overall_split_counts', obj.Handle, ...
                double(numFeatures));
        end

        function counts = forestSplitCounts(obj, sampleIndex, numFeatures)
            %FORESTSPLITCOUNTS Splits on each feature within a single draw.
            counts = mex_stochtree('container_forest_split_counts', obj.Handle, ...
                double(sampleIndex) - 1, double(numFeatures));
        end

        function s = toJson(obj)
            %TOJSON Serialize the stored draws to a JSON string.
            s = mex_stochtree('container_dump_json', obj.Handle);
        end

        function fromJson(obj, s)
            %FROMJSON Replace the stored draws from a JSON string.
            mex_stochtree('container_load_json', obj.Handle, s);
        end

        function saveJson(obj, filename)
            mex_stochtree('container_save_json_file', obj.Handle, filename);
        end

        function loadJson(obj, filename)
            mex_stochtree('container_load_json_file', obj.Handle, filename);
        end
    end
end
