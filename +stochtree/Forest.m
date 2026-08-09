classdef Forest < handle
    %STOCHTREE.FOREST A single "active" tree ensemble.
    %
    %   This is the forest the sampler mutates in place on each sweep. Retained
    %   posterior draws are copied out into a stochtree.ForestContainer.
    %
    %   A newly constructed Forest is "empty": its trees have no leaf values. It
    %   must be initialized either with setRootValue/setRootVector or, more
    %   usually, via stochtree.ForestSampler/initializeForest, which also
    %   propagates the initial values through the sampler's tracking structures.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
        IsEmpty logical = true
    end

    methods
        function obj = Forest(numTrees, outputDimension, isLeafConstant, isExponentiated)
            if nargin < 2 || isempty(outputDimension), outputDimension = 1; end
            if nargin < 3 || isempty(isLeafConstant), isLeafConstant = true; end
            if nargin < 4 || isempty(isExponentiated), isExponentiated = false; end
            obj.Handle = mex_stochtree('forest_create', double(numTrees), ...
                double(outputDimension), logical(isLeafConstant), logical(isExponentiated));
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function n = numTrees(obj)
            n = mex_stochtree('forest_num_trees', obj.Handle);
        end

        function d = outputDimension(obj)
            d = mex_stochtree('forest_output_dimension', obj.Handle);
        end

        function setRootValue(obj, value)
            %SETROOTVALUE Collapse every tree to a root node with a constant leaf.
            mex_stochtree('forest_set_root_value', obj.Handle, double(value));
            obj.IsEmpty = false;
        end

        function setRootVector(obj, values)
            %SETROOTVECTOR Collapse every tree to a root node with a vector leaf.
            mex_stochtree('forest_set_root_vector', obj.Handle, double(values(:)));
            obj.IsEmpty = false;
        end

        function resetRoot(obj)
            %RESETROOT Prune every tree back to its root node.
            mex_stochtree('forest_reset_root', obj.Handle);
        end

        function resetFromContainer(obj, container, sampleIndex)
            %RESETFROMCONTAINER Overwrite this forest with a stored draw.
            %   sampleIndex is 1-based, consistent with MATLAB indexing.
            mex_stochtree('forest_reset_from_container', obj.Handle, ...
                container.Handle, double(sampleIndex) - 1);
            obj.IsEmpty = false;
        end

        function yhat = predict(obj, dataset)
            %PREDICT Return an n-by-1 vector of predictions (basis applied).
            yhat = mex_stochtree('forest_predict', obj.Handle, dataset.Handle);
        end

        function raw = predictRaw(obj, dataset)
            %PREDICTRAW Return n-by-outputDimension raw leaf values (no basis).
            raw = mex_stochtree('forest_predict_raw', obj.Handle, dataset.Handle);
        end

        function markInitialized(obj)
            %MARKINITIALIZED Record that leaves have been set by a sampler.
            obj.IsEmpty = false;
        end

        function n = numLeaves(obj)
            n = mex_stochtree('forest_num_leaves', obj.Handle);
        end

        function s = sumLeafSquared(obj)
            s = mex_stochtree('forest_sum_leaf_squared', obj.Handle);
        end
    end
end
