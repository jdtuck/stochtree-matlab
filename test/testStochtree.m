function tests = testStochtree
%TESTSTOCHTREE Unit tests for the stochtree MATLAB wrapper.
%
%   Run with:  runtests('test/testStochtree.m')
%
%   The statistical tests use loose tolerances and fixed seeds. They check that
%   the sampler recovers signal and calibrates variance roughly correctly, not
%   that it reproduces any exact numbers.

tests = functiontests(localfunctions);
end

%% ---- Fixtures ----------------------------------------------------------

function setupOnce(testCase)
here = fileparts(mfilename('fullpath'));
toolboxRoot = fileparts(here);
addpath(toolboxRoot);
testCase.assertNotEmpty(which('mex_stochtree'), ...
    'mex_stochtree is not on the path. Run build_stochtree first.');
end

function teardown(~)
% Free any C++ objects orphaned by a failed assertion so tests stay isolated.
mex_stochtree('cleanup_all');
end

%% ---- Low-level object tests --------------------------------------------

function testDatasetRoundTrip(testCase)
X = rand(50, 4);
ds = stochtree.Dataset();
ds.addCovariates(X);

testCase.verifyEqual(ds.numRows(), 50);
testCase.verifyEqual(ds.numCovariates(), 4);
testCase.verifyFalse(ds.hasBasis());
% Column-major ordering must survive the trip into C++ and back.
testCase.verifyEqual(ds.getCovariates(), X, 'AbsTol', 1e-12);

W = rand(50, 2);
ds.addBasis(W);
testCase.verifyTrue(ds.hasBasis());
testCase.verifyEqual(ds.numBasis(), 2);
testCase.verifyEqual(ds.getBasis(), W, 'AbsTol', 1e-12);

W2 = rand(50, 2);
ds.updateBasis(W2);
testCase.verifyEqual(ds.getBasis(), W2, 'AbsTol', 1e-12);

w = rand(50, 1) + 0.5;
ds.addWeights(w);
testCase.verifyTrue(ds.hasWeights());
testCase.verifyEqual(ds.getWeights(), w, 'AbsTol', 1e-12);
end

function testResidualArithmetic(testCase)
y = (1:10)';
r = stochtree.Residual(y);
testCase.verifyEqual(r.getValues(), y, 'AbsTol', 1e-12);

r.add(ones(10, 1));
testCase.verifyEqual(r.getValues(), y + 1, 'AbsTol', 1e-12);

r.subtract(2 * ones(10, 1));
testCase.verifyEqual(r.getValues(), y - 1, 'AbsTol', 1e-12);

r.replace(zeros(10, 1));
testCase.verifyEqual(r.getValues(), zeros(10, 1), 'AbsTol', 1e-12);
end

function testHandleLifetime(testCase)
before = mex_stochtree('num_objects');
ds = stochtree.Dataset();
testCase.verifyEqual(mex_stochtree('num_objects'), before + 1);

handle = ds.Handle;
testCase.verifyTrue(mex_stochtree('is_valid', handle));
delete(ds);
testCase.verifyFalse(mex_stochtree('is_valid', handle));

% Deleting twice must not error: destructors can run more than once during
% MATLAB cleanup.
testCase.verifyWarningFree(@() mex_stochtree('delete', handle));
end

function testStaleHandleRejected(testCase)
ds = stochtree.Dataset();
handle = ds.Handle;
delete(ds);
testCase.verifyError(@() mex_stochtree('dataset_num_rows', handle), 'stochtree:handle');
end

function testWrongHandleTypeRejected(testCase)
ds = stochtree.Dataset();
% A Dataset handle passed where a Residual is expected should be caught by the
% type tag rather than reinterpreted as the wrong C++ type.
testCase.verifyError(@() mex_stochtree('residual_get', ds.Handle), 'stochtree:handle');
end

function testUnknownCommandRejected(testCase)
testCase.verifyError(@() mex_stochtree('no_such_command'), 'stochtree:command');
end

%% ---- BART tests ---------------------------------------------------------

function testBartRecoversFriedman(testCase)
rng(7, 'twister');
n = 400;
X = rand(n, 5);
f = 10 * sin(pi * X(:,1) .* X(:,2)) + 20 * (X(:,3) - 0.5).^2 + 10 * X(:,4) + 5 * X(:,5);
y = f + randn(n, 1);

model = stochtree.bart(X, y, 'NumGFR', 10, 'NumMCMC', 100, 'NumTrees', 50, ...
    'RandomSeed', 1);

testCase.verifyEqual(model.NumSamples, 100);
testCase.verifySize(model.YHatTrain, [n, 100]);

yhat = mean(model.YHatTrain, 2);
r2 = 1 - sum((f - yhat).^2) / sum((f - mean(f)).^2);
testCase.verifyGreaterThan(r2, 0.85, ...
    'BART should recover the Friedman function well above this threshold.');

% Posterior mean sigma^2 should land near the true value of 1.
sigma2Hat = mean(model.Sigma2Samples);
testCase.verifyGreaterThan(sigma2Hat, 0.4);
testCase.verifyLessThan(sigma2Hat, 2.5);
end

function testBartPredictMatchesStoredTrainPredictions(testCase)
rng(11);
X = rand(150, 3);
y = X(:,1) * 3 + randn(150, 1) * 0.3;
model = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 20, 'NumTrees', 20, ...
    'RandomSeed', 2);

out = model.predict(X, 'samplesOnly',false);
testCase.verifyEqual(out.yhat, model.YHatTrain, 'AbsTol', 1e-10, ...
    'Re-predicting the training data must reproduce the stored draws exactly.');
end

function testBartWithLeafBasis(testCase)
rng(13);
n = 300;
X = rand(n, 3);
W = randn(n, 1);
% Leaf regression: the outcome is linear in W with an X-dependent slope.
y = W .* (1 + 2 * X(:,1)) + 0.2 * randn(n, 1);

model = stochtree.bart(X, y, 'W', W, 'NumGFR', 10, 'NumMCMC', 50, ...
    'NumTrees', 30, 'RandomSeed', 3);
testCase.verifyTrue(model.HasBasis);
testCase.verifyEqual(model.LeafModel, 1);

yhat = mean(model.predict(X, 'W', W, 'samplesOnly',false).yhat, 2);
r2 = 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);
testCase.verifyGreaterThan(r2, 0.8);
end

function testBartHeteroskedastic(testCase)
rng(17);
n = 600;
X = rand(n, 3);
sigmaX = 0.2 + 1.5 * X(:,1);
y = 2 * X(:,2) + sigmaX .* randn(n, 1);

model = stochtree.bart(X, y, 'NumGFR', 10, 'NumMCMC', 60, 'NumTrees', 50, ...
    'NumTreesVariance', 30, 'RandomSeed', 4);
testCase.verifyTrue(model.IncludeVarianceForest);
testCase.verifySize(model.Sigma2XTrain, [n, model.NumSamples]);

% The estimated variance function should track the truth in rank order.
sigma2Hat = mean(model.Sigma2XTrain, 2);
corrRank = corr(sigma2Hat, sigmaX.^2, 'type', 'Spearman');
testCase.verifyGreaterThan(corrRank, 0.5, ...
    'Variance forest should order observations by true conditional variance.');
end

function testBartTestSetPredictions(testCase)
rng(19);
X = rand(200, 3);
y = X(:,1) * 2 + randn(200, 1) * 0.2;
XTest = rand(50, 3);

model = stochtree.bart(X, y, 'XTest', XTest, 'NumGFR', 5, 'NumMCMC', 20, ...
    'NumTrees', 20, 'RandomSeed', 5);
testCase.verifySize(model.YHatTest, [50, 20]);
testCase.verifyEqual(model.YHatTest, model.predict(XTest, 'samplesOnly',false).yhat, 'AbsTol', 1e-10);
end

function testBartSerializationRoundTrip(testCase)
rng(23);
X = rand(150, 3);
y = X(:,1) * 2 + randn(150, 1) * 0.3;
model = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 20, 'NumTrees', 20, ...
    'RandomSeed', 6);

restored = stochtree.BARTModel.fromStruct(model.toStruct());
testCase.verifyEqual(restored.NumSamples, model.NumSamples);
testCase.verifyEqual(restored.predict(X, 'samplesOnly',false).yhat, model.predict(X, 'samplesOnly',false).yhat, ...
    'AbsTol', 1e-10, 'Serialization must be lossless.');
end

function testBartSeedReproducibility(testCase)
rng(29);
X = rand(120, 3);
y = X(:,1) * 2 + randn(120, 1) * 0.3;

a = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 15, 'NumTrees', 20, 'RandomSeed', 99);
b = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 15, 'NumTrees', 20, 'RandomSeed', 99);
testCase.verifyEqual(b.YHatTrain, a.YHatTrain, 'AbsTol', 1e-12, ...
    'The same seed must give identical draws.');
end

function testBartInputValidation(testCase)
X = rand(50, 3);
y = rand(50, 1);
testCase.verifyError(@() stochtree.bart(X, rand(49, 1)), 'stochtree:size');
testCase.verifyError(@() stochtree.bart([X; nan(1,3)], [y; 1]), 'stochtree:value');

model = stochtree.bart(X, y, 'NumGFR', 2, 'NumMCMC', 5, 'NumTrees', 10, 'RandomSeed', 7);
testCase.verifyError(@() model.predict(rand(10, 5)), 'stochtree:size');
end

%% ---- Multi-chain tests --------------------------------------------------

function testBartMultipleChainsSampleCount(testCase)
rng(101);
X = rand(200, 3);
y = X(:,1) * 2 + randn(200, 1) * 0.3;

model = stochtree.bart(X, y, 'NumGFR', 10, 'NumBurnin', 5, 'NumMCMC', 20, ...
    'NumChains', 4, 'NumTrees', 20, 'RandomSeed', 21);

testCase.verifyEqual(model.NumChains, 4);
testCase.verifyEqual(model.NumSamples, 80, ...
    'Retained draws should be NumChains * NumMCMC.');
testCase.verifySize(model.YHatTrain, [200, 80]);
testCase.verifyNumElements(model.ChainIndex, 80);
% Each chain should contribute exactly NumMCMC draws, in contiguous blocks.
testCase.verifyEqual(sort(unique(model.ChainIndex))', 1:4);
for c = 1:4
    testCase.verifyEqual(sum(model.ChainIndex == c), 20);
end
end

function testBartKeepEveryThinsWithoutLosingDraws(testCase)
rng(103);
X = rand(150, 3);
y = X(:,1) * 2 + randn(150, 1) * 0.3;

% NumMCMC is the retained count; KeepEvery multiplies the iterations run.
model = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 10, 'KeepEvery', 3, ...
    'NumChains', 2, 'NumTrees', 20, 'RandomSeed', 22);
testCase.verifyEqual(model.NumSamples, 20);
end

function testBartKeepGFRRetainsWarmStart(testCase)
rng(105);
X = rand(150, 3);
y = X(:,1) * 2 + randn(150, 1) * 0.3;

model = stochtree.bart(X, y, 'NumGFR', 6, 'NumMCMC', 10, 'NumChains', 2, ...
    'KeepGFR', true, 'NumTrees', 20, 'RandomSeed', 23);
testCase.verifyEqual(model.NumSamples, 6 + 2*10);
% Warm start draws are labelled 0 so they can be told apart from chain draws.
testCase.verifyEqual(sum(model.ChainIndex == 0), 6);
end

function testChainsAreDistinctButAgree(testCase)
rng(107);
n = 400;
X = rand(n, 4);
f = 5*X(:,1) + 3*X(:,2).^2;
y = f + randn(n, 1) * 0.5;

model = stochtree.bart(X, y, 'NumGFR', 10, 'NumBurnin', 50, 'NumMCMC', 60, ...
    'NumChains', 4, 'NumTrees', 50, 'RandomSeed', 24);

% Chains must not be identical -- that would mean the reset failed and every
% chain simply continued the same trajectory.
M = model.chainMatrix(model.Sigma2Samples);
testCase.verifySize(M, [60, 4]);
testCase.verifyGreaterThan(max(std(M, 0, 1)), 0, ...
    'Chains should differ from one another.');

% ...but their posterior means should agree to within sampling noise.
chainMeans = mean(M, 1);
testCase.verifyLessThan(range(chainMeans) / mean(chainMeans), 0.5, ...
    'Chain means for sigma^2 should be broadly consistent.');

% The pooled fit should still be good.
yhat = mean(model.YHatTrain, 2);
r2 = 1 - sum((f - yhat).^2) / sum((f - mean(f)).^2);
testCase.verifyGreaterThan(r2, 0.85);
end

function testChainResetKeepsResidualConsistent(testCase)
% If the residual were left stale after a chain reset, the later chains would
% fit a corrupted target and their predictions would drift away from the
% earlier ones. Compare the first and last chain's posterior means.
rng(109);
n = 400;
X = rand(n, 3);
f = 4*X(:,1) - 2*X(:,2);
y = f + randn(n, 1) * 0.4;

model = stochtree.bart(X, y, 'NumGFR', 8, 'NumBurnin', 50, 'NumMCMC', 50, ...
    'NumChains', 4, 'NumTrees', 40, 'RandomSeed', 25);

first = mean(model.YHatTrain(:, model.ChainIndex == 1), 2);
last = mean(model.YHatTrain(:, model.ChainIndex == 4), 2);
testCase.verifyGreaterThan(corr(first, last), 0.95, ...
    'Later chains should agree with earlier ones after the reset.');
testCase.verifyLessThan(mean(abs(first - last)), 0.5 * std(y));
end

function testChainsFromRootWithoutWarmStart(testCase)
rng(111);
X = rand(300, 3);
y = 3*X(:,1) + randn(300, 1) * 0.4;

% NumGFR = 0 means every chain restarts from a stump instead of a warm start,
% so NumChains is not capped by NumGFR.
model = stochtree.bart(X, y, 'NumGFR', 0, 'NumBurnin', 50, 'NumMCMC', 30, ...
    'NumChains', 3, 'NumTrees', 30, 'RandomSeed', 26);
testCase.verifyEqual(model.NumSamples, 90);
yhat = mean(model.YHatTrain, 2);
testCase.verifyGreaterThan(corr(yhat, y), 0.8);
end

function testTooManyChainsRejected(testCase)
X = rand(100, 3);
y = rand(100, 1);
testCase.verifyError( ...
    @() stochtree.bart(X, y, 'NumGFR', 3, 'NumChains', 5, 'NumMCMC', 5), ...
    'stochtree:value');
testCase.verifyError( ...
    @() stochtree.bart(X, y, 'NumChains', 0, 'NumMCMC', 5), 'stochtree:value');
end

function testBcfMultipleChains(testCase)
rng(113);
n = 600;
X = rand(n, 4);
propensity = 0.3 + 0.4 * X(:,1);
Z = double(rand(n, 1) < propensity);
tau = 1 + X(:,2);
y = 2*X(:,1) + tau .* Z + 0.4 * randn(n, 1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', propensity, ...
    'NumGFR', 10, 'NumBurnin', 50, 'NumMCMC', 50, 'NumChains', 3, ...
    'RandomSeed', 27);

testCase.verifyEqual(model.NumChains, 3);
testCase.verifyEqual(model.NumSamples, 150);
testCase.verifyNumElements(model.B0Samples, 150);
testCase.verifyTrue(all(isfinite(model.B0Samples)));

% The ATE should be consistent across chains, which also confirms the
% adaptive coding state was reset properly at each chain boundary.
ateByChain = arrayfun(@(c) mean(mean(model.TauHatTrain(:, model.ChainIndex == c), 2)), 1:3);
testCase.verifyLessThan(range(ateByChain), 0.5, ...
    'Per-chain ATE estimates should agree.');
testCase.verifyLessThan(abs(mean(ateByChain) - mean(tau)), 0.5);
end

%% ---- R-hat tests ---------------------------------------------------------

function testRhatOnIdenticalChainsIsOne(testCase)
rng(115);
% Four chains drawn from the same distribution: R-hat should sit near 1.
X = randn(500, 4);
R = stochtree.rhat(X);
testCase.verifyGreaterThan(R, 0.99);
testCase.verifyLessThan(R, 1.05);
end

function testRhatDetectsSeparatedChains(testCase)
% Chains centred in different places must be flagged.
X = randn(200, 4) + [0 5 10 15];
R = stochtree.rhat(X);
testCase.verifyGreaterThan(R, 1.5);
end

function testRhatWithChainIndex(testCase)
draws = [randn(100,1); randn(100,1) + 8];
idx = [ones(100,1); 2*ones(100,1)];
R = stochtree.rhat(draws, idx);
testCase.verifyGreaterThan(R, 1.5);

% Warm start draws, labelled 0, must be excluded rather than pooled in.
drawsWithGFR = [zeros(5,1); draws];
idxWithGFR = [zeros(5,1); idx];
testCase.verifyEqual(stochtree.rhat(drawsWithGFR, idxWithGFR), R, 'AbsTol', 1e-12);
end

function testRhatRejectsSingleChain(testCase)
% Both spellings of "one chain" must be rejected, and with the same error,
% since they are the same user mistake.
testCase.verifyError(@() stochtree.rhat(randn(100,1)), 'stochtree:input');
testCase.verifyError(@() stochtree.rhat(randn(10,1), ones(10,1)), 'stochtree:input');
% Two short chains are legitimate, so this must not throw.
testCase.verifyWarningFree(@() stochtree.rhat(randn(20,1), [ones(10,1); 2*ones(10,1)]));
% Chains of different lengths cannot be compared.
testCase.verifyError(@() stochtree.rhat(randn(15,1), [ones(10,1); 2*ones(5,1)]), ...
    'stochtree:value');

end

function testConvergenceDiagnostics(testCase)
rng(117);
X = rand(250, 3);
y = 3*X(:,1) + randn(250, 1) * 0.4;

model = stochtree.bart(X, y, 'NumGFR', 8, 'NumBurnin', 40, 'NumMCMC', 40, ...
    'NumChains', 4, 'NumTrees', 30, 'RandomSeed', 28);
d = model.convergenceDiagnostics();
testCase.verifyEqual(d.numChains, 4);
testCase.verifyTrue(isfield(d, 'sigma2Rhat'));
testCase.verifyTrue(isfinite(d.sigma2Rhat));
testCase.verifyGreaterThan(d.sigma2Rhat, 0.9);

% A single chain has nothing to compare against, and should say so rather
% than returning a meaningless number.
single = stochtree.bart(X, y, 'NumGFR', 5, 'NumMCMC', 20, 'NumTrees', 20, ...
    'RandomSeed', 29);
d1 = single.convergenceDiagnostics();
testCase.verifyTrue(isnan(d1.sigma2Rhat));
testCase.verifyTrue(isfield(d1, 'note'));
end

%% ---- BCF tests ----------------------------------------------------------

function testBcfRecoversHeterogeneousEffect(testCase)
rng(31);
n = 1000;
X = rand(n, 5);
propensity = 0.25 + 0.5 * X(:,1);
Z = double(rand(n, 1) < propensity);
mu = 2 * X(:,1) + X(:,2);
tau = 1 + 2 * X(:,2);
y = mu + tau .* Z + 0.5 * randn(n, 1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', propensity, ...
    'NumGFR', 10, 'NumMCMC', 100, 'RandomSeed', 8);

testCase.verifyEqual(model.NumSamples, 100);
testCase.verifySize(model.TauHatTrain, [n, 100]);

tauHat = mean(model.TauHatTrain, 2);
% Correlation with the true CATE is the substantive check; the level is
% checked separately through the ATE.
testCase.verifyGreaterThan(corr(tauHat, tau), 0.6, ...
    'Estimated CATE should correlate strongly with the truth.');

ate = model.ateSummary();
trueATE = mean(tau);
testCase.verifyLessThan(abs(ate.mean - trueATE), 0.5, ...
    'ATE posterior mean should be close to the true ATE.');
end

function testBcfHomogeneousEffectATECoverage(testCase)
rng(37);
n = 800;
X = rand(n, 4);
propensity = 0.3 + 0.4 * X(:,1);
Z = double(rand(n, 1) < propensity);
y = 2 * X(:,1) + X(:,2) + 1.5 * Z + 0.4 * randn(n, 1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', propensity, ...
    'NumGFR', 10, 'NumMCMC', 100, 'RandomSeed', 9);
ate = model.ateSummary(0.95);
testCase.verifyGreaterThan(1.5, ate.interval(1) - 0.3);
testCase.verifyLessThan(1.5, ate.interval(2) + 0.3);
end

function testBcfInternalPropensity(testCase)
rng(41);
n = 500;
X = rand(n, 4);
Z = double(rand(n, 1) < 0.25 + 0.5 * X(:,1));
y = 2 * X(:,1) + 1.0 * Z + 0.4 * randn(n, 1);

model = stochtree.bcf(X, Z, y, 'NumGFR', 5, 'NumMCMC', 40, 'RandomSeed', 10);
testCase.verifyTrue(model.InternalPropensityModel);
testCase.verifyTrue(model.UsesPropensity);
% Prediction must work without the caller supplying propensity scores, since
% the fitted propensity model is carried along.
out = model.predict(X, Z);
testCase.verifySize(out.tau, [n, model.NumSamples]);
end

function testBcfAdaptiveCodingSamples(testCase)
rng(43);
n = 400;
X = rand(n, 3);
Z = double(rand(n, 1) < 0.5);
y = X(:,1) + Z + 0.3 * randn(n, 1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', repmat(0.5, n, 1), ...
    'NumGFR', 5, 'NumMCMC', 30, 'RandomSeed', 11);
testCase.verifyTrue(model.AdaptiveCoding);
testCase.verifyNumElements(model.B0Samples, 30);
testCase.verifyTrue(all(isfinite(model.B0Samples)));
testCase.verifyTrue(all(isfinite(model.B1Samples)));
end

function testBcfContinuousTreatmentDisablesAdaptiveCoding(testCase)
rng(47);
n = 400;
X = rand(n, 3);
Z = randn(n, 1);
y = X(:,1) + 0.8 * Z + 0.3 * randn(n, 1);

model = testCase.verifyWarning( ...
    @() stochtree.bcf(X, Z, y, 'PropensityCovariate', 'none', ...
        'NumGFR', 5, 'NumMCMC', 30, 'RandomSeed', 12), ...
    'stochtree:adaptiveCoding');
testCase.verifyFalse(model.AdaptiveCoding);
testCase.verifyFalse(model.BinaryTreatment);
end

function testBcfSerializationRoundTrip(testCase)
rng(53);
n = 300;
X = rand(n, 3);
Z = double(rand(n, 1) < 0.5);
y = X(:,1) + Z + 0.3 * randn(n, 1);
pi_ = repmat(0.5, n, 1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', pi_, 'NumGFR', 5, ...
    'NumMCMC', 20, 'RandomSeed', 13);
restored = stochtree.BCFModel.fromStruct(model.toStruct());
testCase.verifyEqual(restored.predict(X, Z, pi_).tau, ...
    model.predict(X, Z, pi_).tau, 'AbsTol', 1e-10);
end
