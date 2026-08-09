classdef BCFModel < handle
    %STOCHTREE.BCFMODEL Fitted Bayesian Causal Forest returned by stochtree.bcf.
    %
    %   Posterior draws are stored column-wise. TauHatTrain is n-by-NumSamples,
    %   so the posterior mean CATE is mean(model.TauHatTrain, 2) and the ATE
    %   posterior is mean(model.TauHatTrain, 1).
    %
    %   See also STOCHTREE.BCF.

    properties
        ForestContainerMu        % stochtree.ForestContainer (prognostic)
        ForestContainerTau       % stochtree.ForestContainer (treatment effect)
        ForestContainerVariance  % stochtree.ForestContainer or []

        YBar (1,1) double = 0
        YStd (1,1) double = 1
        Sigma2Init (1,1) double = 1

        % Posterior draws
        Sigma2Samples double = []
        LeafScaleMuSamples double = []
        LeafScaleTauSamples double = []
        B0Samples double = []
        B1Samples double = []

        MuHatTrain double = []
        TauHatTrain double = []
        YHatTrain double = []
        MuHatTest double = []
        TauHatTest double = []
        YHatTest double = []
        Sigma2XTrain double = []
        Sigma2XTest double = []

        % Model structure
        NumSamples (1,1) double = 0
        NumCovariates (1,1) double = 0
        NumCovariatesAugmented (1,1) double = 0
        TreatmentDim (1,1) double = 1
        BinaryTreatment (1,1) logical = false
        AdaptiveCoding (1,1) logical = false
        UsesPropensity (1,1) logical = false
        PropensityCovariate char = 'prognostic'
        InternalPropensityModel (1,1) logical = false
        PropensityModel = []
        IncludeVarianceForest (1,1) logical = false
        SampleSigma2Global (1,1) logical = true
        NumGFR (1,1) double = 0
        NumBurnin (1,1) double = 0
        NumMCMC (1,1) double = 0
        NumChains (1,1) double = 1
        ChainIndex double = []
    end

    methods
        function out = predict(obj, X, Z, propensity)
            %PREDICT Posterior draws of mu(x), tau(x) and the conditional mean.
            %
            %   OUT = PREDICT(MODEL, X, Z) returns a struct with fields:
            %       mu      n-by-NumSamples prognostic function draws
            %       tau     n-by-NumSamples CATE draws (Z=1 versus Z=0 for a
            %               binary treatment, per unit of Z otherwise)
            %       yhat    n-by-NumSamples conditional mean draws
            %   All are on the original outcome scale.
            %
            %   OUT = PREDICT(MODEL, X, Z, PROPENSITY) supplies propensity
            %   scores. Required when the model was fit with them, unless the
            %   model estimated them internally, in which case they are
            %   predicted from the stored propensity BART model.
            if nargin < 4, propensity = []; end
            X = stochtree.internal.asMatrix(X, 'X');
            Z = stochtree.internal.asMatrix(Z, 'Z');
            if size(Z, 1) == 1 && size(Z, 2) > 1, Z = Z'; end
            n = size(X, 1);
            if size(X, 2) ~= obj.NumCovariates
                error('stochtree:size', ...
                    'X has %d columns but the model was fit with %d.', ...
                    size(X, 2), obj.NumCovariates);
            end
            if size(Z, 1) ~= n || size(Z, 2) ~= obj.TreatmentDim
                error('stochtree:size', 'Z must be %d-by-%d.', n, obj.TreatmentDim);
            end

            if obj.UsesPropensity
                if isempty(propensity)
                    if obj.InternalPropensityModel && ~isempty(obj.PropensityModel)
                        propensity = obj.PropensityModel.posteriorMean(X);
                    else
                        error('stochtree:input', ...
                            ['This model was fit with propensity scores as a ' ...
                             'covariate, so they must be supplied for prediction.']);
                    end
                end
                propensity = double(propensity);
                if isvector(propensity), propensity = propensity(:); end
                XAug = [X, propensity];
            else
                XAug = X;
            end
            if size(XAug, 2) ~= obj.NumCovariatesAugmented
                error('stochtree:size', ...
                    'Augmented covariate matrix has %d columns, expected %d.', ...
                    size(XAug, 2), obj.NumCovariatesAugmented);
            end

            ds = stochtree.Dataset();
            ds.addCovariates(XAug);
            ds.addBasis(Z);   % placeholder; tau draws are read raw

            muRaw = obj.ForestContainerMu.predict(ds);              % n x S
            tauRawArray = obj.ForestContainerTau.predictRaw(ds);    % n x d x S
            S = obj.NumSamples;

            if obj.TreatmentDim == 1
                tauRaw = reshape(tauRawArray, n, S);
                if obj.AdaptiveCoding
                    codingDiff = (obj.B1Samples(:) - obj.B0Samples(:))';
                    tauScaled = tauRaw .* codingDiff;
                    basis = (1 - Z) * obj.B0Samples(:)' + Z * obj.B1Samples(:)';
                else
                    tauScaled = tauRaw;
                    basis = repmat(Z, 1, S);
                end
                treatmentTerm = basis .* tauRaw;
            else
                % Multivariate treatment: the CATE is a vector per observation,
                % so report the raw draws and build the treatment term by
                % contracting over the treatment dimension.
                tauScaled = tauRawArray;
                treatmentTerm = zeros(n, S);
                for d = 1:obj.TreatmentDim
                    treatmentTerm = treatmentTerm + ...
                        Z(:, d) .* reshape(tauRawArray(:, d, :), n, S);
                end
            end

            out.mu = muRaw * obj.YStd + obj.YBar;
            out.tau = tauScaled * obj.YStd;
            out.yhat = (muRaw + treatmentTerm) * obj.YStd + obj.YBar;

            if obj.IncludeVarianceForest
                sigma2Raw = obj.ForestContainerVariance.predict(ds);
                if obj.SampleSigma2Global
                    out.sigma2x = sigma2Raw .* obj.Sigma2Samples(:)';
                else
                    out.sigma2x = sigma2Raw * obj.Sigma2Init * obj.YStd^2;
                end
            end
        end

        function draws = ateSamples(obj)
            %ATESAMPLES Posterior draws of the average treatment effect.
            %   Averages the training-set CATE draws over observations.
            if isempty(obj.TauHatTrain)
                error('stochtree:state', 'No training CATE draws are stored.');
            end
            draws = mean(obj.TauHatTrain, 1)';
        end

        function s = ateSummary(obj, level)
            %ATESUMMARY Posterior mean and credible interval for the ATE.
            if nargin < 2 || isempty(level), level = 0.95; end
            draws = obj.ateSamples();
            tail = (1 - level) / 2;
            s.mean = mean(draws);
            s.median = median(draws);
            s.std = std(draws);
            s.interval = quantile(draws, [tail, 1 - tail]);
            s.level = level;
            if nargout == 0
                fprintf('ATE posterior mean %.4f (sd %.4f), %.0f%% CI [%.4f, %.4f]\n', ...
                    s.mean, s.std, 100 * level, s.interval(1), s.interval(2));
                clear s
            end
        end

        function cate = posteriorCATE(obj, X, Z, propensity)
            %POSTERIORCATE Posterior mean conditional average treatment effect.
            if nargin < 4, propensity = []; end
            p = obj.predict(X, Z, propensity);
            cate = mean(p.tau, 2);
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
                if ~isempty(obj.TauHatTrain)
                    d.ateRhat = stochtree.rhat(obj.ateSamples(), obj.ChainIndex);
                    nRows = size(obj.TauHatTrain, 1);
                    probe = unique(round(linspace(1, nRows, min(50, nRows))));
                    rs = arrayfun(@(i) stochtree.rhat(obj.TauHatTrain(i, :)', ...
                        obj.ChainIndex), probe);
                    d.cateRhatMax = max(rs);
                    d.cateRhatMedian = median(rs);
                end
            else
                d.sigma2Rhat = NaN;
            end
            if nargout == 0
                disp(d);
                clear d
            end
        end

        function s = toStruct(obj)
            %TOSTRUCT Serialize to a plain struct that MATLAB can save/load.
            s = struct();
            props = properties(obj);
            for i = 1:numel(props)
                name = props{i};
                switch name
                    case 'ForestContainerMu'
                        s.ForestContainerMuJson = obj.ForestContainerMu.toJson();
                        s.NumTreesMu = obj.ForestContainerMu.numTrees();
                    case 'ForestContainerTau'
                        s.ForestContainerTauJson = obj.ForestContainerTau.toJson();
                        s.NumTreesTau = obj.ForestContainerTau.numTrees();
                        s.OutputDimTau = obj.ForestContainerTau.outputDimension();
                    case 'ForestContainerVariance'
                        if isempty(obj.ForestContainerVariance)
                            s.ForestContainerVarianceJson = '';
                            s.NumTreesVariance = 0;
                        else
                            s.ForestContainerVarianceJson = obj.ForestContainerVariance.toJson();
                            s.NumTreesVariance = obj.ForestContainerVariance.numTrees();
                        end
                    case 'PropensityModel'
                        if isempty(obj.PropensityModel)
                            s.PropensityModelStruct = [];
                        else
                            s.PropensityModelStruct = obj.PropensityModel.toStruct();
                        end
                    otherwise
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
            %FROMSTRUCT Rebuild a BCFModel from the output of toStruct.
            obj = stochtree.BCFModel();
            skip = {'ForestContainerMuJson', 'NumTreesMu', 'ForestContainerTauJson', ...
                'NumTreesTau', 'OutputDimTau', 'ForestContainerVarianceJson', ...
                'NumTreesVariance', 'PropensityModelStruct'};
            names = fieldnames(s);
            for i = 1:numel(names)
                if any(strcmp(names{i}, skip)), continue; end
                if isprop(obj, names{i})
                    obj.(names{i}) = s.(names{i});
                end
            end
            obj.ForestContainerMu = stochtree.ForestContainer(s.NumTreesMu, 1, true, false);
            obj.ForestContainerMu.fromJson(s.ForestContainerMuJson);
            obj.ForestContainerTau = stochtree.ForestContainer( ...
                s.NumTreesTau, s.OutputDimTau, false, false);
            obj.ForestContainerTau.fromJson(s.ForestContainerTauJson);
            if s.NumTreesVariance > 0
                obj.ForestContainerVariance = stochtree.ForestContainer( ...
                    s.NumTreesVariance, 1, true, true);
                obj.ForestContainerVariance.fromJson(s.ForestContainerVarianceJson);
            end
            if ~isempty(s.PropensityModelStruct)
                obj.PropensityModel = stochtree.BARTModel.fromStruct(s.PropensityModelStruct);
            end
        end

        function obj = load(filename)
            %LOAD Read a model written by BCFModel/save.
            data = load(filename, 'modelStruct');
            obj = stochtree.BCFModel.fromStruct(data.modelStruct);
        end
    end
end
