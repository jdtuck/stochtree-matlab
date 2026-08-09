function model = bcf(X, Z, y, varargin)
%STOCHTREE.BCF Bayesian Causal Forests for heterogeneous treatment effects.
%
%   MODEL = STOCHTREE.BCF(X, Z, Y) fits the Hahn, Murray and Carvalho (2020)
%   causal forest model and returns a stochtree.BCFModel. X is n-by-p
%   covariates, Z is a length-n treatment (binary or continuous), and Y is a
%   length-n outcome.
%
%   The model decomposes the conditional mean into a prognostic term and a
%   treatment term:
%
%       E[Y | X, Z] = mu(X) + b_Z(X),   b_Z(X) = (b_1*Z + b_0*(1-Z)) * tau(X)
%
%   Separating the two lets the treatment forest use a tighter prior than the
%   prognostic forest, which is what protects against regularization-induced
%   confounding. The conditional average treatment effect of Z=1 versus Z=0 is
%   (b_1 - b_0) * tau(X) under adaptive coding, and tau(X) otherwise.
%
%   Data options
%     'XTest', 'ZTest'      Test set covariates and treatment.
%     'PropensityTrain'     Length-n propensity scores. Estimated internally
%                           with a BART model when omitted and needed.
%     'PropensityTest'      Test set propensity scores.
%     'PropensityCovariate' Which forests see the propensity score:
%                           'prognostic' (default), 'treatment_effect',
%                           'both', or 'none'.
%     'FeatureTypes'        Length-p codes, as in stochtree.bart.
%     'CategoricalIndices'  Columns to treat as unordered categorical.
%     'SampleWeights'       Length-n observation variance weights.
%
%   Sampler options
%     'NumGFR'      Default 5.      'NumBurnin'  Default 0.
%     'NumMCMC'     Retained draws per chain. Default 100.
%     'KeepGFR'     Default false.  'KeepBurnin' Default false.
%     'KeepEvery'   Thinning interval. NumMCMC*KeepEvery iterations run and
%                   NumMCMC are retained, per chain. Default 1.
%     'NumChains'   Independent MCMC chains, each warm started from its own
%                   grow-from-root draw. Cannot exceed NumGFR unless NumGFR is
%                   0. Total retained draws are NumChains*NumMCMC. Default 1.
%     'RandomSeed'  Seeds both the C++ RNG and MATLAB's RNG (the adaptive
%                   coding step draws from the latter). Default [].
%     'NumThreads'  Default 1.      'Verbose'    Default false.
%     'Standardize' Default true.   'CutpointGridSize'  Default 100.
%
%   Prognostic forest options
%     'NumTreesMu'   Default 250.   'AlphaMu'  Default 0.95.
%     'BetaMu'       Default 2.     'MinSamplesLeafMu'  Default 5.
%     'MaxDepthMu'   Default 10.    'SampleSigma2LeafMu'  Default true.
%     'Sigma2LeafMuInit'  Default 2*var(resid)/NumTreesMu.
%
%   Treatment effect forest options
%     'NumTreesTau'  Default 100.   'AlphaTau'  Default 0.25.
%     'BetaTau'      Default 3.     'MinSamplesLeafTau'  Default 5.
%     'MaxDepthTau'  Default 5.     'SampleSigma2LeafTau'  Default false.
%     'Sigma2LeafTauInit'  Default 0.5*var(resid)/NumTreesTau.
%
%     The tau forest deliberately gets a much more aggressive depth penalty
%     (alpha 0.25, beta 3, max depth 5) than the mu forest. Treatment effect
%     surfaces are usually far smoother than prognostic surfaces, and this is
%     the prior that encodes that belief.
%
%   Global variance and adaptive coding
%     'SampleSigma2Global'  Default true.
%     'Sigma2Init'          Default var(resid).
%     'Sigma2GlobalShape', 'Sigma2GlobalScale'  Defaults 0 and 0.
%     'AdaptiveCoding'      Learn b_0 and b_1 rather than coding Z as (0,1).
%                           Only used for binary treatments. Default true.
%     'B0Init', 'B1Init'    Defaults -0.5 and 0.5.
%
%   Variance forest options
%     'NumTreesVariance'    Set > 0 for heteroskedastic errors. Default 0.
%
%   Example:
%       n = 1000; X = rand(n, 5);
%       pi = 0.25 + 0.5*X(:,1);
%       Z = double(rand(n,1) < pi);
%       mu = 2*X(:,1) + X(:,2);  tau = 1 + 2*X(:,2);
%       y = mu + tau.*Z + 0.5*randn(n,1);
%       model = stochtree.bcf(X, Z, y, 'PropensityTrain', pi, 'RandomSeed', 1);
%       tauHat = mean(model.TauHatTrain, 2);
%
%   See also STOCHTREE.BART, STOCHTREE.BCFMODEL.

%   Port of the Python stochtree.BCFModel.sample and R stochtree::bcf.

opts = iParseInputs(varargin{:});

X = stochtree.internal.asMatrix(X, 'X');
y = double(y(:));
Z = stochtree.internal.asMatrix(Z, 'Z');
if size(Z, 1) == 1 && size(Z, 2) > 1, Z = Z'; end
[n, p] = size(X);
if numel(y) ~= n
    error('stochtree:size', 'X has %d rows but y has %d elements.', n, numel(y));
end
if size(Z, 1) ~= n
    error('stochtree:size', 'X and Z must have the same number of rows.');
end
stochtree.internal.checkFinite(X, 'X');
stochtree.internal.checkFinite(y, 'y');
stochtree.internal.checkFinite(Z, 'Z');

treatmentDim = size(Z, 2);
multivariateTreatment = treatmentDim > 1;
uniqueZ = unique(Z(:));
binaryTreatment = ~multivariateTreatment && numel(uniqueZ) == 2 && ...
    all(ismember(uniqueZ, [0 1]));

adaptiveCoding = opts.AdaptiveCoding && binaryTreatment;
if opts.AdaptiveCoding && ~binaryTreatment
    warning('stochtree:adaptiveCoding', ...
        'AdaptiveCoding requires a binary 0/1 treatment; disabling it.');
end

if ~isempty(opts.RandomSeed)
    rng(opts.RandomSeed);
end

%% ---- Propensity score -------------------------------------------------
propensityCovariate = validatestring(opts.PropensityCovariate, ...
    {'none', 'prognostic', 'treatment_effect', 'both'}, 'stochtree.bcf', ...
    'PropensityCovariate');

propensityTrain = opts.PropensityTrain;
propensityTest = opts.PropensityTest;
internalPropensityModel = false;
propensityModel = [];

if multivariateTreatment && ~strcmp(propensityCovariate, 'none') && isempty(propensityTrain)
    warning('stochtree:propensity', ...
        ['No propensity scores supplied for a multivariate treatment and none ' ...
         'can be estimated internally; setting PropensityCovariate to ''none''.']);
    propensityCovariate = 'none';
end

if ~strcmp(propensityCovariate, 'none') && isempty(propensityTrain)
    % Estimate the propensity score with a small BART model, matching the
    % Python package's internal default.
    internalPropensityModel = true;
    propensityModel = stochtree.bart(X, Z(:, 1), ...
        'NumGFR', 10, 'NumBurnin', 0, 'NumMCMC', 10, ...
        'NumTrees', 50, 'RandomSeed', opts.RandomSeed, 'Verbose', false);
    propensityTrain = mean(propensityModel.YHatTrain, 2);
    if ~isempty(opts.XTest)
        propensityTest = propensityModel.posteriorMean( ...
            stochtree.internal.asMatrix(opts.XTest, 'XTest'));
    end
end
if ~isempty(propensityTrain)
    propensityTrain = double(propensityTrain);
    if isvector(propensityTrain), propensityTrain = propensityTrain(:); end
    if size(propensityTrain, 1) ~= n
        error('stochtree:size', 'PropensityTrain must have %d rows.', n);
    end
end

% Augment the covariates with the propensity score. Both forests share one
% covariate matrix; which forest may split on the propensity column is
% controlled through the variable weights below.
if strcmp(propensityCovariate, 'none') || isempty(propensityTrain)
    XAug = X;
    propensityCols = [];
else
    XAug = [X, propensityTrain];
    propensityCols = (p + 1):size(XAug, 2);
end
pAug = size(XAug, 2);

featureTypes = zeros(pAug, 1);
if ~isempty(opts.FeatureTypes)
    if numel(opts.FeatureTypes) ~= p
        error('stochtree:size', 'FeatureTypes must have one entry per column of X.');
    end
    featureTypes(1:p) = double(opts.FeatureTypes(:));
end
if ~isempty(opts.CategoricalIndices)
    featureTypes(opts.CategoricalIndices) = 2;
end

variableWeightsMu = repmat(1 / pAug, pAug, 1);
variableWeightsTau = repmat(1 / pAug, pAug, 1);
if ~isempty(propensityCols)
    if strcmp(propensityCovariate, 'treatment_effect')
        variableWeightsMu(propensityCols) = 0;
    elseif strcmp(propensityCovariate, 'prognostic')
        variableWeightsTau(propensityCols) = 0;
    end
end

%% ---- Standardization and prior calibration ----------------------------
if opts.Standardize
    yBar = mean(y);
    yStd = std(y, 1);
    if yStd == 0
        error('stochtree:value', 'y has zero variance; nothing to fit.');
    end
else
    yBar = 0;
    yStd = 1;
end
residTrain = (y - yBar) / yStd;
residVar = var(residTrain, 1);

sigma2Init = opts.Sigma2Init;
if isempty(sigma2Init), sigma2Init = residVar; end
currentSigma2 = sigma2Init;

numTreesMu = opts.NumTreesMu;
numTreesTau = opts.NumTreesTau;

bLeafMu = opts.Sigma2LeafMuScale;
if isempty(bLeafMu), bLeafMu = residVar / numTreesMu; end
bLeafTau = opts.Sigma2LeafTauScale;
if isempty(bLeafTau), bLeafTau = residVar / (2 * numTreesTau); end

currentLeafScaleMu = opts.Sigma2LeafMuInit;
if isempty(currentLeafScaleMu), currentLeafScaleMu = 2 * residVar / numTreesMu; end
sigma2LeafTau = opts.Sigma2LeafTauInit;
if isempty(sigma2LeafTau), sigma2LeafTau = 0.5 * residVar / numTreesTau; end
if multivariateTreatment
    currentLeafScaleTau = eye(treatmentDim) * sigma2LeafTau;
else
    currentLeafScaleTau = sigma2LeafTau;
end

initialLeafScaleMu = currentLeafScaleMu;
initialLeafScaleTau = currentLeafScaleTau;

sampleSigma2LeafTau = opts.SampleSigma2LeafTau && ~multivariateTreatment;

includeVarianceForest = opts.NumTreesVariance > 0;
if includeVarianceForest
    a0 = opts.LeafPriorCalibrationParam;
    aForest = opts.NumTreesVariance / a0^2 + 0.5;
    bForest = opts.NumTreesVariance / a0^2;
    varianceForestLeafInit = 0.6 * residVar;
else
    aForest = 1.0;
    bForest = 1.0;
    varianceForestLeafInit = [];
end

%% ---- Treatment basis --------------------------------------------------
currentB0 = opts.B0Init;
currentB1 = opts.B1Init;
if adaptiveCoding
    tauBasis = (1 - Z) * currentB0 + Z * currentB1;
else
    tauBasis = Z;
end

if multivariateTreatment
    leafModelTau = 2;
    outputDimTau = treatmentDim;
else
    leafModelTau = 1;
    outputDimTau = 1;
end

%% ---- Build the C++ objects -------------------------------------------
seed = opts.RandomSeed;
if isempty(seed), seed = -1; end

dataset = stochtree.Dataset();
dataset.addCovariates(XAug);
dataset.addBasis(tauBasis);
if ~isempty(opts.SampleWeights)
    dataset.addWeights(double(opts.SampleWeights(:)));
end

residual = stochtree.Residual(residTrain);
cppRng = stochtree.RNG(seed);

containerMu = stochtree.ForestContainer(numTreesMu, 1, true, false);
activeForestMu = stochtree.Forest(numTreesMu, 1, true, false);
samplerMu = stochtree.ForestSampler(dataset, featureTypes, numTreesMu, n, ...
    opts.AlphaMu, opts.BetaMu, opts.MinSamplesLeafMu, opts.MaxDepthMu);

containerTau = stochtree.ForestContainer(numTreesTau, outputDimTau, false, false);
activeForestTau = stochtree.Forest(numTreesTau, outputDimTau, false, false);
samplerTau = stochtree.ForestSampler(dataset, featureTypes, numTreesTau, n, ...
    opts.AlphaTau, opts.BetaTau, opts.MinSamplesLeafTau, opts.MaxDepthTau);

% The mu forest must be initialized first: it is a constant-leaf model and its
% initialization subtracts the initial prediction from the residual, which the
% tau forest's initialization then builds on.
samplerMu.initializeForest(dataset, residual, activeForestMu, 0, mean(residTrain));
if multivariateTreatment
    samplerTau.initializeForest(dataset, residual, activeForestTau, leafModelTau, ...
        zeros(treatmentDim, 1));
else
    samplerTau.initializeForest(dataset, residual, activeForestTau, leafModelTau, 0.0);
end

if includeVarianceForest
    containerVar = stochtree.ForestContainer(opts.NumTreesVariance, 1, true, true);
    activeForestVar = stochtree.Forest(opts.NumTreesVariance, 1, true, true);
    samplerVar = stochtree.ForestSampler(dataset, featureTypes, opts.NumTreesVariance, ...
        n, 0.95, 2.0, 5, 10);
    samplerVar.initializeForest(dataset, residual, activeForestVar, 3, varianceForestLeafInit);
else
    containerVar = [];
    activeForestVar = [];
    samplerVar = [];
end

globalVarModel = stochtree.GlobalVarianceModel();
leafVarModelMu = stochtree.LeafVarianceModel();
leafVarModelTau = stochtree.LeafVarianceModel();

%% ---- Sampling ---------------------------------------------------------
numGFR = opts.NumGFR;
numBurnin = opts.NumBurnin;
numMCMC = opts.NumMCMC;
numChains = opts.NumChains;
keepEvery = opts.KeepEvery;
if numGFR + numBurnin + numMCMC == 0
    error('stochtree:value', 'NumGFR + NumBurnin + NumMCMC must be positive.');
end
if numChains < 1 || mod(numChains, 1) ~= 0
    error('stochtree:value', 'NumChains must be a positive integer.');
end
if numGFR > 0 && numChains > numGFR
    error('stochtree:value', ...
        ['NumChains (%d) cannot exceed NumGFR (%d). Each chain warm starts from ' ...
         'its own grow-from-root draw. Raise NumGFR, lower NumChains, or set ' ...
         'NumGFR to 0 to start every chain from the root instead.'], ...
        numChains, numGFR);
end

% Every warm start draw is retained up front and pruned afterwards if KeepGFR
% is false, so that chain c can restart from draw numGFR-c+1 by index.
perChain = opts.KeepBurnin * numBurnin + numMCMC;
numRetained = numGFR + numChains * perChain;

globalVarSamples = nan(numRetained, 1);
leafScaleMuSamples = nan(numRetained, 1);
leafScaleTauSamples = nan(numRetained, 1);
b0Samples = nan(numRetained, 1);
b1Samples = nan(numRetained, 1);
chainIndex = zeros(numRetained, 1);    % 0 marks a shared warm start draw
sampleCounter = 0;
currentChain = 0;

muOpts = struct('cutpointGridSize', opts.CutpointGridSize, ...
    'leafModelScale', currentLeafScaleMu, 'variableWeights', variableWeightsMu, ...
    'aForest', aForest, 'bForest', bForest, 'globalVariance', currentSigma2, ...
    'leafModel', 0, 'numFeaturesSubsample', pAug, 'keepForest', true, ...
    'gfr', true, 'numThreads', opts.NumThreads);
tauOpts = muOpts;
tauOpts.leafModel = leafModelTau;
tauOpts.variableWeights = variableWeightsTau;
tauOpts.leafModelScale = currentLeafScaleTau;
varOpts = muOpts;
varOpts.leafModel = 3;
varOpts.leafModelScale = 1.0;

% Stage 1: one grow-from-root warm start, shared by every chain.
for iter = 1:numGFR
    iSweep(true, true);
    if opts.Verbose && mod(iter, 25) == 0
        fprintf('stochtree.bcf: warm start %d of %d (sigma^2 = %.4f)\n', ...
            iter, numGFR, currentSigma2);
    end
end

% Stage 2: independent MCMC chains, each restarted from its own warm start.
numChainIters = numBurnin + numMCMC * keepEvery;
for chain = 1:numChains
    currentChain = chain;
    iResetChain(chain);
    for iter = 1:numChainIters
        if iter <= numBurnin
            keepSample = opts.KeepBurnin;
        else
            keepSample = mod(iter - numBurnin, keepEvery) == 0;
        end
        iSweep(false, keepSample);
        if opts.Verbose && mod(iter, 25) == 0
            fprintf('stochtree.bcf: chain %d/%d, iteration %d/%d (sigma^2 = %.4f)\n', ...
                chain, numChains, iter, numChainIters, currentSigma2);
        end
    end
end

if numGFR > 0 && ~opts.KeepGFR
    for i = 1:numGFR
        containerMu.deleteSample(1);
        containerTau.deleteSample(1);
        if includeVarianceForest, containerVar.deleteSample(1); end
    end
    keepIdx = (numGFR + 1):numRetained;
    globalVarSamples = globalVarSamples(keepIdx);
    leafScaleMuSamples = leafScaleMuSamples(keepIdx);
    leafScaleTauSamples = leafScaleTauSamples(keepIdx);
    b0Samples = b0Samples(keepIdx);
    b1Samples = b1Samples(keepIdx);
    chainIndex = chainIndex(keepIdx);
end

%% ---- Assemble the model ----------------------------------------------
model = stochtree.BCFModel();
model.ForestContainerMu = containerMu;
model.ForestContainerTau = containerTau;
model.ForestContainerVariance = containerVar;
model.YBar = yBar;
model.YStd = yStd;
model.Sigma2Init = sigma2Init;
model.NumSamples = containerMu.numSamples();
model.NumCovariates = p;
model.NumCovariatesAugmented = pAug;
model.TreatmentDim = treatmentDim;
model.BinaryTreatment = binaryTreatment;
model.AdaptiveCoding = adaptiveCoding;
model.PropensityCovariate = propensityCovariate;
model.UsesPropensity = ~isempty(propensityCols);
model.InternalPropensityModel = internalPropensityModel;
model.PropensityModel = propensityModel;
model.IncludeVarianceForest = includeVarianceForest;
model.SampleSigma2Global = opts.SampleSigma2Global;
model.Sigma2Samples = globalVarSamples * yStd^2;
model.LeafScaleMuSamples = leafScaleMuSamples;
model.LeafScaleTauSamples = leafScaleTauSamples;
model.B0Samples = b0Samples;
model.B1Samples = b1Samples;
model.NumGFR = numGFR;
model.NumBurnin = numBurnin;
model.NumMCMC = numMCMC;
model.NumChains = numChains;
model.ChainIndex = chainIndex;

trainPred = model.predict(X, Z, propensityTrain);
model.MuHatTrain = trainPred.mu;
model.TauHatTrain = trainPred.tau;
model.YHatTrain = trainPred.yhat;
if includeVarianceForest
    model.Sigma2XTrain = trainPred.sigma2x;
end

if ~isempty(opts.XTest)
    XTest = stochtree.internal.asMatrix(opts.XTest, 'XTest');
    if isempty(opts.ZTest)
        error('stochtree:input', 'ZTest is required when XTest is supplied.');
    end
    testPred = model.predict(XTest, opts.ZTest, propensityTest);
    model.MuHatTest = testPred.mu;
    model.TauHatTest = testPred.tau;
    model.YHatTest = testPred.yhat;
    if includeVarianceForest
        model.Sigma2XTest = testPred.sigma2x;
    end
end

%% ---- Nested helpers ---------------------------------------------------
    function iSweep(isGFR, keepSample)
        % One full Gibbs sweep: prognostic forest, variance parameters,
        % treatment forest, adaptive coding, then the variance forest.
        if keepSample
            sampleCounter = sampleCounter + 1;
            chainIndex(sampleCounter) = currentChain;
        end

        % --- Prognostic forest
        muOpts.gfr = isGFR;
        muOpts.keepForest = keepSample;
        muOpts.globalVariance = currentSigma2;
        muOpts.leafModelScale = currentLeafScaleMu;
        samplerMu.sampleOneIteration(containerMu, activeForestMu, dataset, ...
            residual, cppRng, muOpts);

        if opts.SampleSigma2Global
            currentSigma2 = globalVarModel.sample(residual, cppRng, ...
                opts.Sigma2GlobalShape, opts.Sigma2GlobalScale);
        end
        if opts.SampleSigma2LeafMu
            currentLeafScaleMu = leafVarModelMu.sample(activeForestMu, cppRng, ...
                opts.Sigma2LeafMuShape, bLeafMu);
            if keepSample
                leafScaleMuSamples(sampleCounter) = currentLeafScaleMu;
            end
        end

        % --- Treatment effect forest
        tauOpts.gfr = isGFR;
        tauOpts.keepForest = keepSample;
        tauOpts.globalVariance = currentSigma2;
        tauOpts.leafModelScale = currentLeafScaleTau;
        samplerTau.sampleOneIteration(containerTau, activeForestTau, dataset, ...
            residual, cppRng, tauOpts);

        % --- Adaptive coding parameters b_0 and b_1
        if adaptiveCoding
            muX = activeForestMu.predictRaw(dataset);
            tauX = activeForestTau.predictRaw(dataset);
            % Regress the outcome net of the prognostic term on tau(x)
            % separately within the treated and control arms. This uses the
            % standardized outcome, not the running residual, which already
            % has both forests subtracted out.
            partialResid = residTrain - muX;
            isControl = Z(:, 1) == 0;
            isTreated = ~isControl;

            sTT0 = sum(tauX(isControl).^2);
            sTT1 = sum(tauX(isTreated).^2);
            sTY0 = sum(tauX(isControl) .* partialResid(isControl));
            sTY1 = sum(tauX(isTreated) .* partialResid(isTreated));

            % N(0, 1/2) priors on b_0 and b_1 give these full conditionals.
            currentB0 = (sTY0 / (sTT0 + 2 * currentSigma2)) + ...
                sqrt(currentSigma2 / (sTT0 + 2 * currentSigma2)) * randn();
            currentB1 = (sTY1 / (sTT1 + 2 * currentSigma2)) + ...
                sqrt(currentSigma2 / (sTT1 + 2 * currentSigma2)) * randn();

            iApplyCoding();

            if keepSample
                b0Samples(sampleCounter) = currentB0;
                b1Samples(sampleCounter) = currentB1;
            end
        end

        if sampleSigma2LeafTau
            currentLeafScaleTau = leafVarModelTau.sample(activeForestTau, cppRng, ...
                opts.Sigma2LeafTauShape, bLeafTau);
            if keepSample
                leafScaleTauSamples(sampleCounter) = currentLeafScaleTau;
            end
        end

        % --- Variance forest
        if includeVarianceForest
            varOpts.gfr = isGFR;
            varOpts.keepForest = keepSample;
            varOpts.globalVariance = currentSigma2;
            samplerVar.sampleOneIteration(containerVar, activeForestVar, dataset, ...
                residual, cppRng, varOpts);
        end

        if opts.SampleSigma2Global && keepSample
            globalVarSamples(sampleCounter) = currentSigma2;
        end
    end

    function iApplyCoding()
        % Rebuild the treatment basis from the current b_0 and b_1 and push it
        % through the tau tracker, which is stale until the residual is
        % recomputed against the new basis.
        tauBasis = (1 - Z) * currentB0 + Z * currentB1;
        dataset.updateBasis(tauBasis);
        samplerTau.propagateBasisUpdate(dataset, residual, activeForestTau);
    end

    function iResetChain(chain)
        % Put the sampler back into a warm start state before running a chain.
        %
        % reconstitute() repairs the residual incrementally: it adds back the
        % tracker's cached predictions and subtracts the new forest's. Both
        % forests share one residual, so each is swapped in and reconstituted
        % in turn, and the adaptive coding basis is reapplied last, once the
        % tau tracker already matches the restored forest.
        if numGFR > 0
            forestInd = numGFR - chain + 1;

            activeForestMu.resetFromContainer(containerMu, forestInd);
            samplerMu.reconstitute(activeForestMu, dataset, residual, true);

            activeForestTau.resetFromContainer(containerTau, forestInd);
            samplerTau.reconstitute(activeForestTau, dataset, residual, true);

            if includeVarianceForest
                activeForestVar.resetFromContainer(containerVar, forestInd);
                samplerVar.reconstitute(activeForestVar, dataset, residual, false);
            end

            if opts.SampleSigma2Global
                currentSigma2 = globalVarSamples(forestInd);
            end
            if opts.SampleSigma2LeafMu
                currentLeafScaleMu = leafScaleMuSamples(forestInd);
            end
            if sampleSigma2LeafTau
                currentLeafScaleTau = leafScaleTauSamples(forestInd);
            end
            if adaptiveCoding
                currentB0 = b0Samples(forestInd);
                currentB1 = b1Samples(forestInd);
                iApplyCoding();
            end
        else
            % No warm start: every chain starts from stumps.
            activeForestMu.resetRoot();
            activeForestMu.setRootValue(mean(residTrain) / numTreesMu);
            samplerMu.reconstitute(activeForestMu, dataset, residual, true);

            activeForestTau.resetRoot();
            if multivariateTreatment
                activeForestTau.setRootVector(zeros(treatmentDim, 1));
            else
                activeForestTau.setRootValue(0);
            end
            samplerTau.reconstitute(activeForestTau, dataset, residual, true);

            if includeVarianceForest
                activeForestVar.resetRoot();
                activeForestVar.setRootValue( ...
                    log(varianceForestLeafInit) / opts.NumTreesVariance);
                samplerVar.reconstitute(activeForestVar, dataset, residual, false);
            end

            currentSigma2 = sigma2Init;
            currentLeafScaleMu = initialLeafScaleMu;
            currentLeafScaleTau = initialLeafScaleTau;
            if adaptiveCoding
                currentB0 = opts.B0Init;
                currentB1 = opts.B1Init;
                iApplyCoding();
            end
        end
    end
end

% -------------------------------------------------------------------------
function opts = iParseInputs(varargin)
q = inputParser;
q.FunctionName = 'stochtree.bcf';

addParameter(q, 'XTest', []);
addParameter(q, 'ZTest', []);
addParameter(q, 'PropensityTrain', []);
addParameter(q, 'PropensityTest', []);
addParameter(q, 'PropensityCovariate', 'prognostic');
addParameter(q, 'FeatureTypes', []);
addParameter(q, 'CategoricalIndices', []);
addParameter(q, 'SampleWeights', []);

addParameter(q, 'NumGFR', 5);
addParameter(q, 'NumBurnin', 0);
addParameter(q, 'NumMCMC', 100);
addParameter(q, 'KeepGFR', false);
addParameter(q, 'KeepBurnin', false);
addParameter(q, 'KeepEvery', 1);
addParameter(q, 'NumChains', 1);
addParameter(q, 'RandomSeed', []);
addParameter(q, 'NumThreads', 1);
addParameter(q, 'CutpointGridSize', 100);
addParameter(q, 'Verbose', false);
addParameter(q, 'Standardize', true);

addParameter(q, 'NumTreesMu', 250);
addParameter(q, 'AlphaMu', 0.95);
addParameter(q, 'BetaMu', 2.0);
addParameter(q, 'MinSamplesLeafMu', 5);
addParameter(q, 'MaxDepthMu', 10);
addParameter(q, 'SampleSigma2LeafMu', true);
addParameter(q, 'Sigma2LeafMuInit', []);
addParameter(q, 'Sigma2LeafMuShape', 3);
addParameter(q, 'Sigma2LeafMuScale', []);

addParameter(q, 'NumTreesTau', 100);
addParameter(q, 'AlphaTau', 0.25);
addParameter(q, 'BetaTau', 3.0);
addParameter(q, 'MinSamplesLeafTau', 5);
addParameter(q, 'MaxDepthTau', 5);
addParameter(q, 'SampleSigma2LeafTau', false);
addParameter(q, 'Sigma2LeafTauInit', []);
addParameter(q, 'Sigma2LeafTauShape', 3);
addParameter(q, 'Sigma2LeafTauScale', []);

addParameter(q, 'SampleSigma2Global', true);
addParameter(q, 'Sigma2Init', []);
addParameter(q, 'Sigma2GlobalShape', 0);
addParameter(q, 'Sigma2GlobalScale', 0);
addParameter(q, 'AdaptiveCoding', true);
addParameter(q, 'B0Init', -0.5);
addParameter(q, 'B1Init', 0.5);

addParameter(q, 'NumTreesVariance', 0);
addParameter(q, 'LeafPriorCalibrationParam', 1.5);

parse(q, varargin{:});
opts = q.Results;
end
