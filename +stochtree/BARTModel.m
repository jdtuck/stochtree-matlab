classdef BARTModel < handle
    %STOCHTREE.BARTMODEL Fitted BART model returned by stochtree.bart.
    %
    %   Posterior draws are stored column-wise: YHatTrain is n-by-NumSamples,
    %   so posterior means are mean(model.YHatTrain, 2) and 95% intervals are
    %   quantile(model.YHatTrain, [0.025 0.975], 2).
    %
    %   See also STOCHTREE.BART.

    properties
        % Fitted forests
        ForestContainerMean      % stochtree.ForestContainer
        ForestContainerVariance  % stochtree.ForestContainer or []

        % Outcome standardization
        YBar (1,1) double = 0
        YStd (1,1) double = 1
        Sigma2Init (1,1) double = 1

        % Posterior draws
        Sigma2Samples double = []      % original outcome scale
        LeafScaleSamples double = []
        YHatTrain double = []
        YHatTest double = []
        Sigma2XTrain double = []
        Sigma2XTest double = []

        % Model structure
        NumSamples (1,1) double = 0
        NumCovariates (1,1) double = 0
        NumBasis (1,1) double = 0
        HasBasis (1,1) logical = false
        LeafModel (1,1) double = 0
        FeatureTypes double = []
        Standardize (1,1) logical = true
        SampleSigma2Global (1,1) logical = true
        SampleSigma2Leaf (1,1) logical = true
        IncludeVarianceForest (1,1) logical = false
        NumGFR (1,1) double = 0
        NumBurnin (1,1) double = 0
        NumMCMC (1,1) double = 0
        NumChains (1,1) double = 1
        ChainIndex double = []
    end

    methods
        function out = predict(obj, X, W)
            %PREDICT Posterior predictive draws of the conditional mean.
            %
            %   OUT = PREDICT(MODEL, X) returns a struct with field yhat, an
            %   n-by-NumSamples matrix of draws on the original outcome scale.
            %   When the model includes a variance forest, OUT also has field
            %   sigma2x with the conditional error variance draws.
            %
            %   OUT = PREDICT(MODEL, X, W) supplies the leaf regression basis
            %   for models fit with the 'W' option.
            if nargin < 3, W = []; end
            X = stochtree.internal.asMatrix(X, 'X');
            if size(X, 2) ~= obj.NumCovariates
                error('stochtree:size', ...
                    'X has %d columns but the model was fit with %d.', ...
                    size(X, 2), obj.NumCovariates);
            end

            ds = stochtree.Dataset();
            ds.addCovariates(X);
            if obj.HasBasis
                if isempty(W)
                    error('stochtree:input', ...
                        'This model was fit with a leaf basis, so W must be supplied.');
                end
                W = stochtree.internal.asMatrix(W, 'W');
                if size(W, 1) ~= size(X, 1) || size(W, 2) ~= obj.NumBasis
                    error('stochtree:size', 'W must be %d-by-%d.', ...
                        size(X, 1), obj.NumBasis);
                end
                ds.addBasis(W);
            end

            raw = obj.ForestContainerMean.predict(ds);
            out.yhat = raw * obj.YStd + obj.YBar;

            if obj.IncludeVarianceForest
                % The variance container is exponentiated, so predict returns
                % the multiplicative variance function directly.
                sigma2Raw = obj.ForestContainerVariance.predict(ds);
                if obj.SampleSigma2Global
                    out.sigma2x = sigma2Raw .* obj.Sigma2Samples(:)';
                else
                    out.sigma2x = sigma2Raw * obj.Sigma2Init * obj.YStd^2;
                end
            end
        end

        function m = posteriorMean(obj, X, W)
            %POSTERIORMEAN Posterior mean prediction, averaged over draws.
            if nargin < 3, W = []; end
            p = obj.predict(X, W);
            m = mean(p.yhat, 2);
        end

        function ci = credibleInterval(obj, X, level, W)
            %CREDIBLEINTERVAL Pointwise posterior interval, n-by-2.
            if nargin < 3 || isempty(level), level = 0.95; end
            if nargin < 4, W = []; end
            p = obj.predict(X, W);
            tail = (1 - level) / 2;
            ci = quantile(p.yhat, [tail, 1 - tail], 2);
        end

        function counts = variableSplitCounts(obj)
            %VARIABLESPLITCOUNTS Splits on each covariate, pooled over draws.
            %   A crude but useful variable importance summary.
            counts = obj.ForestContainerMean.splitCounts(obj.NumCovariates);
        end


        function M = chainMatrix(obj, draws)
            %CHAINMATRIX Reshape a per-draw vector into iterations-by-chains.
            %   Warm start draws (chain label 0) are dropped. Useful for trace
            %   plots: plot(model.chainMatrix(model.Sigma2Samples)).
            draws = draws(:);
            if numel(draws) ~= obj.NumSamples
                error('stochtree:size', ...
                    'draws has %d elements but the model holds %d.', ...
                    numel(draws), obj.NumSamples);
            end
            idx = obj.ChainIndex(:);
            labels = unique(idx(idx > 0));
            if isempty(labels)
                M = draws;
                return
            end
            counts = arrayfun(@(c) sum(idx == c), labels);
            if numel(unique(counts)) ~= 1
                error('stochtree:value', 'Chains have unequal lengths.');
            end
            M = zeros(counts(1), numel(labels));
            for c = 1:numel(labels)
                M(:, c) = draws(idx == labels(c));
            end
        end

        function d = convergenceDiagnostics(obj)
            %CONVERGENCEDIAGNOSTICS Split R-hat for the sampled quantities.
            %   With a single chain there is nothing to compare against, so
            %   the values are NaN and a note explains why.
            d = struct();
            d.numChains = obj.NumChains;
            if obj.NumChains < 2
                d.note = ['R-hat needs at least two chains. Refit with ' ...
                    '''NumChains'', 4 to get a convergence diagnostic.'];
            end
            if obj.NumChains >= 2
                if ~isempty(obj.Sigma2Samples) && obj.SampleSigma2Global
                    d.sigma2Rhat = stochtree.rhat(obj.Sigma2Samples, obj.ChainIndex);
                end
                if ~isempty(obj.LeafScaleSamples) && obj.SampleSigma2Leaf
                    d.leafScaleRhat = stochtree.rhat(obj.LeafScaleSamples, obj.ChainIndex);
                end
                if ~isempty(obj.YHatTrain)
                    % Worst-case R-hat over a spread of fitted values gives a
                    % rough read on whether the forests themselves have mixed.
                    nRows = size(obj.YHatTrain, 1);
                    probe = unique(round(linspace(1, nRows, min(50, nRows))));
                    rs = arrayfun(@(i) stochtree.rhat(obj.YHatTrain(i, :)', ...
                        obj.ChainIndex), probe);
                    d.predictionRhatMax = max(rs);
                    d.predictionRhatMedian = median(rs);
                end
            else
                d.sigma2Rhat = NaN;
            end
            if nargout == 0
                disp(d);
                clear d
            end
        end

        function s = summary(obj)
            %SUMMARY One-line description of the fitted model.
            s = sprintf(['BART: %d retained draws (%d chain(s) x %d MCMC, ' ...
                '%d GFR + %d burnin), %d trees, %d covariates%s'], ...
                obj.NumSamples, obj.NumChains, obj.NumMCMC, obj.NumGFR, ...
                obj.NumBurnin, obj.ForestContainerMean.numTrees(), ...
                obj.NumCovariates, ...
                iIf(obj.IncludeVarianceForest, ', heteroskedastic', ''));
            if nargout == 0
                disp(s);
                if ~isempty(obj.Sigma2Samples)
                    fprintf('  posterior mean sigma^2 = %.4f\n', mean(obj.Sigma2Samples));
                end
                clear s
            end
        end

        function s = toStruct(obj)
            %TOSTRUCT Serialize to a plain struct that MATLAB can save/load.
            %
            %   The C++ forests are serialized to JSON strings, so the result
            %   contains no handles and survives save/load and parfor.
            s = struct();
            props = properties(obj);
            for i = 1:numel(props)
                name = props{i};
                if strcmp(name, 'ForestContainerMean')
                    s.ForestContainerMeanJson = obj.ForestContainerMean.toJson();
                    s.NumTreesMean = obj.ForestContainerMean.numTrees();
                    s.OutputDimensionMean = obj.ForestContainerMean.outputDimension();
                elseif strcmp(name, 'ForestContainerVariance')
                    if isempty(obj.ForestContainerVariance)
                        s.ForestContainerVarianceJson = '';
                        s.NumTreesVariance = 0;
                    else
                        s.ForestContainerVarianceJson = obj.ForestContainerVariance.toJson();
                        s.NumTreesVariance = obj.ForestContainerVariance.numTrees();
                    end
                else
                    s.(name) = obj.(name);
                end
            end
        end

        function save(obj, filename)
            %SAVE Write the model to a .mat file.
            modelStruct = obj.toStruct(); %#ok<NASGU>
            save(filename, 'modelStruct', '-v7.3');
        end
    end

    methods (Static)
        function obj = fromStruct(s)
            %FROMSTRUCT Rebuild a BARTModel from the output of toStruct.
            obj = stochtree.BARTModel();
            names = fieldnames(s);
            for i = 1:numel(names)
                name = names{i};
                if any(strcmp(name, {'ForestContainerMeanJson', 'NumTreesMean', ...
                        'OutputDimensionMean', 'ForestContainerVarianceJson', ...
                        'NumTreesVariance'}))
                    continue
                end
                if isprop(obj, name)
                    obj.(name) = s.(name);
                end
            end
            leafConstant = ~s.HasBasis;
            obj.ForestContainerMean = stochtree.ForestContainer( ...
                s.NumTreesMean, s.OutputDimensionMean, leafConstant, false);
            obj.ForestContainerMean.fromJson(s.ForestContainerMeanJson);
            if s.NumTreesVariance > 0
                obj.ForestContainerVariance = stochtree.ForestContainer( ...
                    s.NumTreesVariance, 1, true, true);
                obj.ForestContainerVariance.fromJson(s.ForestContainerVarianceJson);
            end
        end

        function obj = load(filename)
            %LOAD Read a model written by BARTModel/save.
            data = load(filename, 'modelStruct');
            obj = stochtree.BARTModel.fromStruct(data.modelStruct);
        end
    end
end

function out = iIf(cond, a, b)
if cond, out = a; else, out = b; end
end
