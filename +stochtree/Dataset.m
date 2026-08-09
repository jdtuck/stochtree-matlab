classdef Dataset < handle
    %STOCHTREE.DATASET Covariates, optional leaf basis and variance weights.
    %
    %   Thin MATLAB handle around the C++ StochTree::ForestDataset class. The
    %   underlying C++ object copies whatever you pass in, so the MATLAB arrays
    %   you supply are free to go out of scope afterwards.
    %
    %   Example:
    %       ds = stochtree.Dataset();
    %       ds.addCovariates(X);
    %
    %   See also STOCHTREE.BART, STOCHTREE.FORESTSAMPLER.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = Dataset()
            obj.Handle = mex_stochtree('dataset_create');
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function addCovariates(obj, X)
            %ADDCOVARIATES Load an n-by-p matrix of covariates.
            mex_stochtree('dataset_add_covariates', obj.Handle, ...
                stochtree.internal.asMatrix(X, 'X'));
        end

        function addBasis(obj, W)
            %ADDBASIS Load an n-by-k leaf regression basis.
            mex_stochtree('dataset_add_basis', obj.Handle, ...
                stochtree.internal.asMatrix(W, 'W'));
        end

        function updateBasis(obj, W)
            %UPDATEBASIS Overwrite the leaf regression basis in place.
            mex_stochtree('dataset_update_basis', obj.Handle, ...
                stochtree.internal.asMatrix(W, 'W'));
        end

        function addWeights(obj, w)
            %ADDWEIGHTS Load a length-n vector of observation variance weights.
            mex_stochtree('dataset_add_weights', obj.Handle, double(w(:)));
        end

        function updateWeights(obj, w, exponentiate)
            %UPDATEWEIGHTS Overwrite the variance weights in place.
            if nargin < 3, exponentiate = false; end
            mex_stochtree('dataset_update_weights', obj.Handle, double(w(:)), ...
                logical(exponentiate));
        end

        function n = numRows(obj)
            n = mex_stochtree('dataset_num_rows', obj.Handle);
        end

        function p = numCovariates(obj)
            p = mex_stochtree('dataset_num_covariates', obj.Handle);
        end

        function k = numBasis(obj)
            k = mex_stochtree('dataset_num_basis', obj.Handle);
        end

        function tf = hasBasis(obj)
            tf = mex_stochtree('dataset_has_basis', obj.Handle);
        end

        function tf = hasWeights(obj)
            tf = mex_stochtree('dataset_has_weights', obj.Handle);
        end

        function X = getCovariates(obj)
            X = mex_stochtree('dataset_get_covariates', obj.Handle);
        end

        function W = getBasis(obj)
            W = mex_stochtree('dataset_get_basis', obj.Handle);
        end

        function w = getWeights(obj)
            w = mex_stochtree('dataset_get_weights', obj.Handle);
        end
    end
end
