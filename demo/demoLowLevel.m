%% Driving the sampler directly
% The high-level stochtree.bart is a convenience layer. When you need a custom
% model -- a different residual update, an extra Gibbs step, a nonstandard
% prior schedule -- you can drive the same primitives yourself. This script
% reimplements plain BART in about 40 lines.

clear;
rng(3);

n = 400; p = 5; numTrees = 50;
X = rand(n, p);
y = 10*sin(pi*X(:,1).*X(:,2)) + 10*X(:,4) + randn(n, 1);

% Standardize, as stochtree.bart does internally
yBar = mean(y);  yStd = std(y, 1);
resid = (y - yBar) / yStd;
residVar = var(resid, 1);

% Build the C++ objects
ds = stochtree.Dataset();
ds.addCovariates(X);
residual = stochtree.Residual(resid);
cppRng = stochtree.RNG(1234);

container = stochtree.ForestContainer(numTrees, 1, true, false);
forest = stochtree.Forest(numTrees, 1, true, false);
featureTypes = zeros(p, 1);          % all numeric
sampler = stochtree.ForestSampler(ds, featureTypes, numTrees, n, 0.95, 2.0, 5, 10);

% Constant-leaf Gaussian model (code 0); this also primes the tracker
sampler.initializeForest(ds, residual, forest, 0, mean(resid));

globalVar = stochtree.GlobalVarianceModel();
leafVar = stochtree.LeafVarianceModel();

sigma2 = residVar;
leafScale = residVar / numTrees;
opts = struct('cutpointGridSize', 100, 'leafModelScale', leafScale, ...
    'variableWeights', repmat(1/p, p, 1), 'aForest', 1, 'bForest', 1, ...
    'globalVariance', sigma2, 'leafModel', 0, 'numFeaturesSubsample', p, ...
    'keepForest', true, 'gfr', true, 'numThreads', 1);

numGFR = 10; numMCMC = 200;
sigma2Draws = nan(numGFR + numMCMC, 1);

for iter = 1:(numGFR + numMCMC)
    opts.gfr = iter <= numGFR;
    opts.globalVariance = sigma2;
    opts.leafModelScale = leafScale;
    sampler.sampleOneIteration(container, forest, ds, residual, cppRng, opts);

    sigma2 = globalVar.sample(residual, cppRng, 0, 0);
    leafScale = leafVar.sample(forest, cppRng, 3, residVar/numTrees);
    sigma2Draws(iter) = sigma2;
end

% Discard the warm start draws
for i = 1:numGFR
    container.deleteSample(1);
end

preds = container.predict(ds) * yStd + yBar;
yhat = mean(preds, 2);
fprintf('Draws retained: %d\n', container.numSamples());
fprintf('Train RMSE %.4f, R^2 %.4f\n', sqrt(mean((y-yhat).^2)), ...
    1 - sum((y-yhat).^2)/sum((y-mean(y)).^2));
fprintf('Posterior mean sigma: %.4f (true 1.0)\n', ...
    sqrt(mean(sigma2Draws(numGFR+1:end))) * yStd);
