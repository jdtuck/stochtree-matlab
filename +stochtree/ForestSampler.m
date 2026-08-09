classdef ForestSampler < handle
    %STOCHTREE.FORESTSAMPLER Tree prior plus the sampler's tracking structures.
    %
    %   One ForestSampler is needed per forest in a model: BART with a variance
    %   forest needs two, BCF needs two or three. Samplers may share a Dataset,
    %   but each keeps its own ForestTracker.
    %
    %   Leaf model codes (matching the C++ StochTree::ModelType enum):
    %       0  constant leaf Gaussian
    %       1  univariate regression leaf Gaussian
    %       2  multivariate regression leaf Gaussian
    %       3  log-linear variance
    %       4  cloglog ordinal
    %
    %   Feature type codes:
    %       0  numeric   1  ordered categorical   2  unordered categorical

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
        FeatureTypes double
        NumTrees double
    end

    methods
        function obj = ForestSampler(dataset, featureTypes, numTrees, numObs, ...
                alpha, beta, minSamplesLeaf, maxDepth)
            if nargin < 5 || isempty(alpha), alpha = 0.95; end
            if nargin < 6 || isempty(beta), beta = 2.0; end
            if nargin < 7 || isempty(minSamplesLeaf), minSamplesLeaf = 5; end
            if nargin < 8 || isempty(maxDepth), maxDepth = 10; end
            obj.FeatureTypes = double(featureTypes(:));
            obj.NumTrees = double(numTrees);
            obj.Handle = mex_stochtree('sampler_create', dataset.Handle, ...
                obj.FeatureTypes, double(numTrees), double(numObs), ...
                double(alpha), double(beta), double(minSamplesLeaf), double(maxDepth));
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function initializeForest(obj, dataset, residual, forest, leafModel, initialValues)
            %INITIALIZEFOREST Set constant leaves and propagate through the tracker.
            %   initialValues is divided by the number of trees internally, so
            %   pass the value you want the whole forest to predict.
            mex_stochtree('sampler_initialize_forest', obj.Handle, dataset.Handle, ...
                residual.Handle, forest.Handle, double(leafModel), ...
                double(initialValues(:)));
            forest.markInitialized();
        end

        function reconstitute(obj, forest, dataset, residual, isMeanModel)
            %RECONSTITUTE Rebuild the tracker around an existing forest.
            mex_stochtree('sampler_reconstitute', obj.Handle, forest.Handle, ...
                dataset.Handle, residual.Handle, logical(isMeanModel));
        end

        function sampleOneIteration(obj, container, forest, dataset, residual, rng, opts)
            %SAMPLEONEITERATION Run a single GFR or MCMC sweep over the forest.
            %
            %   opts is a struct with fields: cutpointGridSize, leafModelScale,
            %   variableWeights, aForest, bForest, globalVariance, leafModel,
            %   numFeaturesSubsample, keepForest, gfr, numThreads, and optionally
            %   sweepUpdateIndices (0-based tree indices; defaults to all trees).
            if isfield(opts, 'sweepUpdateIndices') && ~isempty(opts.sweepUpdateIndices)
                sweep = double(opts.sweepUpdateIndices(:));
            else
                sweep = (0:(obj.NumTrees - 1))';
            end
            mex_stochtree('sampler_sample_one_iteration', obj.Handle, ...
                container.Handle, forest.Handle, dataset.Handle, residual.Handle, ...
                rng.Handle, obj.FeatureTypes, sweep, ...
                double(opts.cutpointGridSize), double(opts.leafModelScale), ...
                double(opts.variableWeights(:)), double(opts.aForest), ...
                double(opts.bForest), double(opts.globalVariance), ...
                double(opts.leafModel), double(opts.numFeaturesSubsample), ...
                logical(opts.keepForest), logical(opts.gfr), double(opts.numThreads));
        end

        function preds = cachedPredictions(obj)
            %CACHEDPREDICTIONS Training predictions computed during the last sweep.
            %   Free to read: the sampler already computed them. Note that for a
            %   leaf regression forest these are premultiplied by the basis.
            preds = mex_stochtree('sampler_cached_predictions', obj.Handle);
        end

        function propagateBasisUpdate(obj, dataset, residual, forest)
            %PROPAGATEBASISUPDATE Refresh the residual after the basis changed.
            mex_stochtree('sampler_propagate_basis_update', obj.Handle, ...
                dataset.Handle, residual.Handle, forest.Handle);
        end

        function propagateResidualUpdate(obj, residual)
            %PROPAGATERESIDUALUPDATE Refresh the tracker after the outcome changed.
            mex_stochtree('sampler_propagate_residual_update', obj.Handle, residual.Handle);
        end

        function adjustResidual(obj, dataset, residual, forest, requiresBasis, add)
            %ADJUSTRESIDUAL Add or subtract a whole forest from the residual.
            mex_stochtree('sampler_adjust_residual', obj.Handle, dataset.Handle, ...
                residual.Handle, forest.Handle, logical(requiresBasis), logical(add));
        end

        function setPrior(obj, alpha, beta, minSamplesLeaf, maxDepth)
            mex_stochtree('sampler_set_prior', obj.Handle, double(alpha), ...
                double(beta), double(minSamplesLeaf), double(maxDepth));
        end

        function s = getPrior(obj)
            s = mex_stochtree('sampler_get_prior', obj.Handle);
        end
    end
end
