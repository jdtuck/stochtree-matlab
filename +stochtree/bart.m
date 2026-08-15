function model = bart(X, y, varargin)
%STOCHTREE.BART Bayesian Additive Regression Trees (BART / XBART).
%
%   MODEL = STOCHTREE.BART(X, Y) fits a BART model to an n-by-p covariate
%   matrix X and a length-n outcome Y, and returns a stochtree.BARTModel.
%
%   MODEL = STOCHTREE.BART(X, Y, 'Name', Value, ...) sets options.
%
%   The sampler runs in two stages, matching the R and Python packages:
%   'NumGFR' iterations of the grow-from-root (XBART) algorithm to warm start,
%   then 'NumBurnin' + 'NumMCMC' iterations of the classic BART MCMC. Setting
%   NumGFR to 0 gives plain BART; setting NumMCMC to 0 gives plain XBART.
%
%   Data options
%     'XTest'          n_test-by-p matrix; predictions are computed and stored.
%     'W'              n-by-k leaf regression basis. With a basis the forest
%                      predicts W(i,:) * leaf(i) instead of a constant.
%     'WTest'          Test set basis, required if W and XTest are both given.
%     'SampleWeights'  Length-n vector of observation variance weights.
%     'FeatureTypes'   Length-p vector of codes: 0 numeric, 1 ordered
%                      categorical, 2 unordered categorical. Default all zeros.
%     'CategoricalIndices'  Column indices to mark as unordered categorical.
%                      Convenience alternative to FeatureTypes.
%
%   Sampler options
%     'NumGFR'         Grow-from-root warm start iterations. Default 5.
%     'NumBurnin'      MCMC burn-in iterations. Default 0.
%     'NumMCMC'        Retained MCMC draws per chain. Default 100.
%     'KeepGFR'        Retain the warm start draws. Default false.
%     'KeepBurnin'     Retain the burn-in draws. Default false.
%     'KeepEvery'      Thinning interval. NumMCMC*KeepEvery iterations run and
%                      NumMCMC are retained, per chain. Default 1.
%     'NumChains'      Independent MCMC chains, each warm started from its own
%                      grow-from-root draw. Cannot exceed NumGFR unless NumGFR
%                      is 0, in which case every chain starts from a stump.
%                      Total retained draws are NumChains*NumMCMC. Default 1.
%     'RandomSeed'     Seed for the C++ RNG. Default [] (nondeterministic).
%     'NumThreads'     OpenMP threads. Default 1. Use -1 for all available.
%     'CutpointGridSize'  Candidate cutpoints per feature in GFR. Default 100.
%     'Verbose'        Print progress. Default false.
%
%   Mean forest options
%     'NumTrees'          Default 200.
%     'Alpha', 'Beta'     Tree prior split probability alpha*(1+d)^-beta.
%                         Defaults 0.95 and 2.
%     'MinSamplesLeaf'    Default 5.
%     'MaxDepth'          Default 10. Use -1 for unbounded.
%     'SampleSigma2Leaf'  Sample the leaf scale. Default true (ignored when a
%                         multivariate basis is supplied).
%     'Sigma2LeafInit'    Default var(resid)/NumTrees.
%     'Sigma2LeafShape'   Default 3.
%     'Sigma2LeafScale'   Default var(resid)/NumTrees.
%     'NumFeaturesSubsample'  Features considered per GFR split. Default p.
%
%   Global variance options
%     'Standardize'         Center and scale Y before sampling. Default true.
%     'SampleSigma2Global'  Default true.
%     'Sigma2Init'          Default var(resid).
%     'Sigma2GlobalShape'   Default 0.
%     'Sigma2GlobalScale'   Default 0.
%     'VariableWeights'     Length-p relative split probabilities. Default
%                           uniform.
%
%   Variance forest options (heteroskedastic BART, off by default)
%     'NumTreesVariance'    Set > 0 to model sigma^2(x). Default 0.
%     'VarianceForestAlpha', 'VarianceForestBeta'  Defaults 0.95 and 2.
%     'VarianceForestMinSamplesLeaf'  Default 5.
%     'VarianceForestMaxDepth'        Default 10.
%     'LeafPriorCalibrationParam'     Default 1.5.
%     'VarianceForestLeafInit'        Default 0.6*var(resid).
%     'VarianceForestShape', 'VarianceForestScale'  Calibrated from
%                           LeafPriorCalibrationParam when not supplied.
%
%   Example:
%       n = 500; X = rand(n, 5);
%       y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + randn(n,1);
%       model = stochtree.bart(X, y, 'NumMCMC', 200, 'RandomSeed', 1);
%       yhat = mean(model.predict(X), 2);
%
%   See also STOCHTREE.BCF, STOCHTREE.BARTMODEL.

%   Port of the Python stochtree.BARTModel.sample and R stochtree::bart.

opts = iParseInputs(X, y, varargin{:});

X = stochtree.internal.asMatrix(X, 'X');
y = double(y(:));
[n, p] = size(X);
if numel(y) ~= n
    error('stochtree:size', 'X has %d rows but y has %d elements.', n, numel(y));
end
stochtree.internal.checkFinite(X, 'X');
stochtree.internal.checkFinite(y, 'y');

%% ---- Feature types and variable weights -------------------------------
featureTypes = zeros(p, 1);
if ~isempty(opts.FeatureTypes)
    if numel(opts.FeatureTypes) ~= p
        error('stochtree:size', 'FeatureTypes must have one entry per column of X.');
    end
    featureTypes = double(opts.FeatureTypes(:));
end
if ~isempty(opts.CategoricalIndices)
    featureTypes(opts.CategoricalIndices) = 2;
end

if isempty(opts.VariableWeights)
    variableWeights = repmat(1 / p, p, 1);
else
    variableWeights = double(opts.VariableWeights(:));
    if numel(variableWeights) ~= p
        error('stochtree:size', 'VariableWeights must have one entry per column of X.');
    end
    if any(variableWeights < 0)
        error('stochtree:value', 'VariableWeights cannot be negative.');
    end
end

numFeaturesSubsample = opts.NumFeaturesSubsample;
if isempty(numFeaturesSubsample), numFeaturesSubsample = p; end

%% ---- Basis ------------------------------------------------------------
hasBasis = ~isempty(opts.W);
if hasBasis
    W = stochtree.internal.asMatrix(opts.W, 'W');
    if size(W, 1) ~= n
        error('stochtree:size', 'W must have the same number of rows as X.');
    end
    numBasis = size(W, 2);
    if numBasis == 1
        leafModel = 1;   % univariate regression leaf
        outputDim = 1;
    else
        leafModel = 2;   % multivariate regression leaf
        outputDim = numBasis;
    end
    leafConstant = false;
else
    W = [];
    numBasis = 0;
    leafModel = 0;       % constant leaf
    outputDim = 1;
    leafConstant = true;
end

%% ---- Standardization and prior calibration ----------------------------
if opts.Standardize
    yBar = mean(y);
    yStd = std(y, 1);          % population std, matching numpy's default
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

aGlobal = opts.Sigma2GlobalShape;
bGlobal = opts.Sigma2GlobalScale;

aLeaf = opts.Sigma2LeafShape;
bLeaf = opts.Sigma2LeafScale;
if isempty(bLeaf), bLeaf = residVar / opts.NumTrees; end

if isempty(opts.Sigma2LeafInit)
    leafScaleDiag = residVar / opts.NumTrees;
else
    leafScaleDiag = opts.Sigma2LeafInit;
end
if leafModel == 2
    currentLeafScale = eye(numBasis) * leafScaleDiag;
else
    currentLeafScale = leafScaleDiag;
end

initialLeafScale = currentLeafScale;

% The leaf variance Gibbs step is only defined for scalar leaf scales.
sampleSigma2Leaf = opts.SampleSigma2Leaf && leafModel ~= 2;

includeVarianceForest = opts.NumTreesVariance > 0;
if includeVarianceForest
    a0 = opts.LeafPriorCalibrationParam;
    aForest = opts.VarianceForestShape;
    bForest = opts.VarianceForestScale;
    if isempty(aForest), aForest = opts.NumTreesVariance / a0^2 + 0.5; end
    if isempty(bForest), bForest = opts.NumTreesVariance / a0^2; end
    varianceForestLeafInit = opts.VarianceForestLeafInit;
    if isempty(varianceForestLeafInit), varianceForestLeafInit = 0.6 * residVar; end
else
    aForest = 1.0;
    bForest = 1.0;
    varianceForestLeafInit = [];
end

%% ---- Build the C++ objects -------------------------------------------
seed = opts.RandomSeed;
if isempty(seed), seed = -1; end

datasetTrain = stochtree.Dataset();
datasetTrain.addCovariates(X);
if hasBasis, datasetTrain.addBasis(W); end
if ~isempty(opts.SampleWeights)
    datasetTrain.addWeights(double(opts.SampleWeights(:)));
end

residual = stochtree.Residual(residTrain);
cppRng = stochtree.RNG(seed);

forestContainerMean = stochtree.ForestContainer(opts.NumTrees, outputDim, leafConstant, false);
activeForestMean = stochtree.Forest(opts.NumTrees, outputDim, leafConstant, false);
samplerMean = stochtree.ForestSampler(datasetTrain, featureTypes, opts.NumTrees, n, ...
    opts.Alpha, opts.Beta, opts.MinSamplesLeaf, opts.MaxDepth);

initValMean = mean(residTrain);
if leafModel == 2
    samplerMean.initializeForest(datasetTrain, residual, activeForestMean, leafModel, ...
        repmat(initValMean, numBasis, 1));
else
    samplerMean.initializeForest(datasetTrain, residual, activeForestMean, leafModel, initValMean);
end

if includeVarianceForest
    forestContainerVar = stochtree.ForestContainer(opts.NumTreesVariance, 1, true, true);
    activeForestVar = stochtree.Forest(opts.NumTreesVariance, 1, true, true);
    samplerVar = stochtree.ForestSampler(datasetTrain, featureTypes, ...
        opts.NumTreesVariance, n, opts.VarianceForestAlpha, opts.VarianceForestBeta, ...
        opts.VarianceForestMinSamplesLeaf, opts.VarianceForestMaxDepth);
    samplerVar.initializeForest(datasetTrain, residual, activeForestVar, 3, ...
        varianceForestLeafInit);
else
    forestContainerVar = [];
    activeForestVar = [];
    samplerVar = [];
end

globalVarModel = stochtree.GlobalVarianceModel();
leafVarModel = stochtree.LeafVarianceModel();

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
% is false. Leaving them in the container while the chains run lets chain c
% warm start from draw numGFR-c+1 by index.
perChain = opts.KeepBurnin * numBurnin + numMCMC;
numRetained = numGFR + numChains * perChain;

globalVarSamples = nan(numRetained, 1);
leafScaleSamples = nan(numRetained, 1);
chainIndex = zeros(numRetained, 1);    % 0 marks a shared warm start draw
sampleCounter = 0;
currentChain = 0;

meanOpts = struct( ...
    'cutpointGridSize', opts.CutpointGridSize, ...
    'leafModelScale', currentLeafScale, ...
    'variableWeights', variableWeights, ...
    'aForest', aForest, ...
    'bForest', bForest, ...
    'globalVariance', currentSigma2, ...
    'leafModel', leafModel, ...
    'numFeaturesSubsample', numFeaturesSubsample, ...
    'keepForest', true, ...
    'gfr', true, ...
    'numThreads', opts.NumThreads);

varOpts = meanOpts;
varOpts.leafModel = 3;
varOpts.leafModelScale = 1.0;

% Stage 1: one grow-from-root warm start, shared by every chain.
for iter = 1:numGFR
    iSweep(true, true);
    if opts.Verbose && mod(iter, 25) == 0
        fprintf('stochtree.bart: warm start %d of %d (sigma^2 = %.4f)\n', ...
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
            fprintf('stochtree.bart: chain %d/%d, iteration %d/%d (sigma^2 = %.4f)\n', ...
                chain, numChains, iter, numChainIters, currentSigma2);
        end
    end
end

% Drop the warm start draws unless the caller asked to keep them.
if numGFR > 0 && ~opts.KeepGFR
    for i = 1:numGFR
        forestContainerMean.deleteSample(1);
        if includeVarianceForest
            forestContainerVar.deleteSample(1);
        end
    end
    keepIdx = (numGFR + 1):numRetained;
    globalVarSamples = globalVarSamples(keepIdx);
    leafScaleSamples = leafScaleSamples(keepIdx);
    chainIndex = chainIndex(keepIdx);
end
numSamples = forestContainerMean.numSamples();

%% ---- Assemble the model ----------------------------------------------
model = stochtree.BARTModel();
model.ForestContainerMean = forestContainerMean;
model.ForestContainerVariance = forestContainerVar;
model.YBar = yBar;
model.YStd = yStd;
model.Sigma2Init = sigma2Init;
model.NumSamples = numSamples;
model.NumCovariates = p;
model.NumBasis = numBasis;
model.HasBasis = hasBasis;
model.LeafModel = leafModel;
model.FeatureTypes = featureTypes;
model.Standardize = opts.Standardize;
model.SampleSigma2Global = opts.SampleSigma2Global;
model.SampleSigma2Leaf = sampleSigma2Leaf;
model.IncludeVarianceForest = includeVarianceForest;
model.NumGFR = numGFR;
model.NumBurnin = numBurnin;
model.NumMCMC = numMCMC;
model.NumChains = numChains;
model.ChainIndex = chainIndex;

% Global variance draws are reported on the original outcome scale.
model.Sigma2Samples = globalVarSamples * yStd^2;
model.samples.s2 = model.Sigma2Samples;
model.LeafScaleSamples = leafScaleSamples;

trainPred = model.predict(X, 'W', W, 'samplesOnly',false);
model.YHatTrain = trainPred.yhat;
if includeVarianceForest
    model.Sigma2XTrain = trainPred.sigma2x;
end

if ~isempty(opts.XTest)
    XTest = stochtree.internal.asMatrix(opts.XTest, 'XTest');
    if size(XTest, 2) ~= p
        error('stochtree:size', 'XTest must have the same number of columns as X.');
    end
    WTest = opts.WTest;
    if hasBasis && isempty(WTest)
        error('stochtree:input', 'WTest is required when W and XTest are both supplied.');
    end
    testPred = model.predict(XTest, 'W', WTest, 'samplesOnly',false);
    model.YHatTest = testPred.yhat;
    if includeVarianceForest
        model.Sigma2XTest = testPred.sigma2x;
    end
end

%% ---- Nested helpers ---------------------------------------------------
    function iSweep(isGFR, keepSample)
        % One sweep over the mean forest, the variance forest and the
        % variance parameters. Nested so it can share the sampler state
        % rather than threading a dozen handles through an argument list.
        if keepSample
            sampleCounter = sampleCounter + 1;
            chainIndex(sampleCounter) = currentChain;
        end

        meanOpts.gfr = isGFR;
        meanOpts.keepForest = keepSample;
        meanOpts.globalVariance = currentSigma2;
        meanOpts.leafModelScale = currentLeafScale;
        samplerMean.sampleOneIteration(forestContainerMean, activeForestMean, ...
            datasetTrain, residual, cppRng, meanOpts);

        if includeVarianceForest
            varOpts.gfr = isGFR;
            varOpts.keepForest = keepSample;
            varOpts.globalVariance = currentSigma2;
            samplerVar.sampleOneIteration(forestContainerVar, activeForestVar, ...
                datasetTrain, residual, cppRng, varOpts);
        end

        if opts.SampleSigma2Global
            currentSigma2 = globalVarModel.sample(residual, cppRng, aGlobal, bGlobal);
            if keepSample
                globalVarSamples(sampleCounter) = currentSigma2;
            end
        end
        if sampleSigma2Leaf
            currentLeafScale = leafVarModel.sample(activeForestMean, cppRng, aLeaf, bLeaf);
            if keepSample
                leafScaleSamples(sampleCounter) = currentLeafScale;
            end
        end
    end

    function iResetChain(chain)
        % Put the sampler back into a warm start state before running a chain.
        %
        % reconstitute() repairs the residual incrementally: it adds back the
        % tracker's cached predictions and subtracts the new forest's, so the
        % residual stays consistent without ever being recomputed from y. That
        % is why the forests must be swapped in before the tracker is rebuilt.
        if numGFR > 0
            % Chains walk backwards through the warm start draws so that
            % chain 1 continues from where the warm start left off.
            forestInd = numGFR - chain + 1;
            activeForestMean.resetFromContainer(forestContainerMean, forestInd);
            samplerMean.reconstitute(activeForestMean, datasetTrain, residual, true);
            if sampleSigma2Leaf
                currentLeafScale = leafScaleSamples(forestInd);
            end
            if opts.SampleSigma2Global
                currentSigma2 = globalVarSamples(forestInd);
            end
            if includeVarianceForest
                activeForestVar.resetFromContainer(forestContainerVar, forestInd);
                samplerVar.reconstitute(activeForestVar, datasetTrain, residual, false);
            end
        else
            % No warm start: every chain starts from a stump at the prior mean.
            activeForestMean.resetRoot();
            if leafModel == 2
                activeForestMean.setRootVector( ...
                    repmat(initValMean / opts.NumTrees, numBasis, 1));
            else
                activeForestMean.setRootValue(initValMean / opts.NumTrees);
            end
            samplerMean.reconstitute(activeForestMean, datasetTrain, residual, true);
            currentSigma2 = sigma2Init;
            currentLeafScale = initialLeafScale;
            if includeVarianceForest
                activeForestVar.resetRoot();
                activeForestVar.setRootValue( ...
                    log(varianceForestLeafInit) / opts.NumTreesVariance);
                samplerVar.reconstitute(activeForestVar, datasetTrain, residual, false);
            end
        end
    end
end

% -------------------------------------------------------------------------
function opts = iParseInputs(X, y, varargin)
q = inputParser;
q.FunctionName = 'stochtree.bart';

addParameter(q, 'XTest', []);
addParameter(q, 'W', []);
addParameter(q, 'WTest', []);
addParameter(q, 'SampleWeights', []);
addParameter(q, 'FeatureTypes', []);
addParameter(q, 'CategoricalIndices', []);

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

addParameter(q, 'NumTrees', 200);
addParameter(q, 'Alpha', 0.95);
addParameter(q, 'Beta', 2.0);
addParameter(q, 'MinSamplesLeaf', 5);
addParameter(q, 'MaxDepth', 10);
addParameter(q, 'SampleSigma2Leaf', true);
addParameter(q, 'Sigma2LeafInit', []);
addParameter(q, 'Sigma2LeafShape', 3);
addParameter(q, 'Sigma2LeafScale', []);
addParameter(q, 'NumFeaturesSubsample', []);

addParameter(q, 'Standardize', true);
addParameter(q, 'SampleSigma2Global', true);
addParameter(q, 'Sigma2Init', []);
addParameter(q, 'Sigma2GlobalShape', 0);
addParameter(q, 'Sigma2GlobalScale', 0);
addParameter(q, 'VariableWeights', []);

addParameter(q, 'NumTreesVariance', 0);
addParameter(q, 'VarianceForestAlpha', 0.95);
addParameter(q, 'VarianceForestBeta', 2.0);
addParameter(q, 'VarianceForestMinSamplesLeaf', 5);
addParameter(q, 'VarianceForestMaxDepth', 10);
addParameter(q, 'LeafPriorCalibrationParam', 1.5);
addParameter(q, 'VarianceForestLeafInit', []);
addParameter(q, 'VarianceForestShape', []);
addParameter(q, 'VarianceForestScale', []);

parse(q, varargin{:});
opts = q.Results;

if opts.KeepEvery < 1
    error('stochtree:value', 'KeepEvery must be a positive integer.');
end
if opts.NumTrees < 1
    error('stochtree:value', 'NumTrees must be positive.');
end
end
