classdef testLowLevel < matlab.unittest.TestCase
    %TESTLOWLEVEL Tests for the low-level handle classes and the sampling loop.
    %
    %   These exercise the C++ primitives directly: Forest, ForestContainer,
    %   ForestSampler, RNG, GlobalVarianceModel and LeafVarianceModel. They
    %   complement testStochtree.m, which covers Dataset and Residual, and the
    %   high-level bart/bcf entry points.
    %
    %   Run with:  runtests('test/testLowLevel.m')

    methods (TestClassSetup)
        function requireGateway(testCase)
            here = fileparts(mfilename('fullpath'));
            toolboxRoot = fileparts(here);
            addpath(toolboxRoot);
            testCase.assertNotEmpty(which('mex_stochtree'), ...
                'mex_stochtree is not on the path. Run build_stochtree first.');
        end
    end

    methods (TestMethodTeardown)
        function freeOrphans(~)
            % Free any C++ objects orphaned by a failed assertion so tests
            % stay isolated from one another.
            mex_stochtree('cleanup_all');
        end
    end

    %% ---- RNG ------------------------------------------------------------

    methods (Test)
        function rngCreatesAndFrees(testCase)
            before = mex_stochtree('num_objects');
            rngObj = stochtree.RNG(42);
            testCase.verifyEqual(mex_stochtree('num_objects'), before + 1);
            testCase.verifyTrue(mex_stochtree('is_valid', rngObj.Handle));
            delete(rngObj);
        end

        function rngDefaultSeedIsNonDeterministic(testCase)
            % No argument seeds from random_device; the object should still be
            % valid. We only assert it constructs.
            rngObj = stochtree.RNG();
            testCase.verifyTrue(mex_stochtree('is_valid', rngObj.Handle));
        end

        %% ---- Forest -----------------------------------------------------

        function forestConstructionDefaults(testCase)
            f = stochtree.Forest(10);
            testCase.verifyEqual(f.numTrees(), 10);
            testCase.verifyEqual(f.outputDimension(), 1);
            % A fresh forest reports itself empty until leaves are set.
            testCase.verifyTrue(f.IsEmpty);
        end

        function forestSetRootValueMakesConstantPrediction(testCase)
            n = 20;
            X = rand(n, 3);
            ds = stochtree.Dataset();
            ds.addCovariates(X);

            f = stochtree.Forest(5, 1, true, false);
            f.setRootValue(0.4);
            testCase.verifyFalse(f.IsEmpty);
            % Five trees each holding 0.4 predicts 2.0 everywhere.
            preds = f.predict(ds);
            testCase.verifySize(preds, [n, 1]);
            testCase.verifyEqual(preds, repmat(2.0, n, 1), 'AbsTol', 1e-10);
        end

        function forestResetRootPrunesBackToStump(testCase)
            % After setRootValue then resetRoot the forest is a stump again and
            % predicts zero everywhere. (numLeaves / sumLeafSquared are only
            % meaningful on a sampler-initialized forest, so they are covered in
            % the end-to-end loop below, not on a bare forest.)
            X = rand(15, 2);
            ds = stochtree.Dataset();
            ds.addCovariates(X);
            f = stochtree.Forest(4, 1, true, false);
            f.setRootValue(1.0);
            f.resetRoot();
            testCase.verifyEqual(f.predict(ds), zeros(15, 1), 'AbsTol', 1e-10);
        end

        function forestVectorLeafOutputDimension(testCase)
            f = stochtree.Forest(2, 2, true, false);
            testCase.verifyEqual(f.outputDimension(), 2);
            f.setRootVector([1; 2]);
            testCase.verifyFalse(f.IsEmpty);
        end

        %% ---- ForestContainer --------------------------------------------

        function containerStartsEmpty(testCase)
            c = stochtree.ForestContainer(10, 1, true, false);
            testCase.verifyEqual(c.numSamples(), 0);
            testCase.verifyEqual(c.numTrees(), 10);
            testCase.verifyEqual(c.outputDimension(), 1);
        end

        function containerJsonRoundTrip(testCase)
            % Draw a couple of forests, dump to JSON, reload into a fresh
            % container and check the predictions match exactly.
            [c, ds] = testLowLevel.sampleSmallContainer(2001);
            preds = c.predict(ds);
            json = c.toJson();

            c2 = stochtree.ForestContainer(c.numTrees(), 1, true, false);
            c2.fromJson(json);
            testCase.verifyEqual(c2.numSamples(), c.numSamples());
            testCase.verifyEqual(c2.predict(ds), preds, 'AbsTol', 1e-10, ...
                'JSON serialization of the container must be lossless.');
        end

        function containerSaveLoadJsonFile(testCase)
            [c, ds] = testLowLevel.sampleSmallContainer(2002);
            preds = c.predict(ds);

            f = [tempname, '.json'];
            cleanup = onCleanup(@() testLowLevel.deleteIfPresent(f));
            c.saveJson(f);
            testCase.verifyTrue(isfile(f));

            c2 = stochtree.ForestContainer(c.numTrees(), 1, true, false);
            c2.loadJson(f);
            testCase.verifyEqual(c2.predict(ds), preds, 'AbsTol', 1e-10);
        end

        function containerDeleteSampleReducesCount(testCase)
            [c, ~] = testLowLevel.sampleSmallContainer(2003);
            before = c.numSamples();
            testCase.assertGreaterThan(before, 1);
            c.deleteSample(1);
            testCase.verifyEqual(c.numSamples(), before - 1);
        end

        function containerPredictRawShape(testCase)
            [c, ds] = testLowLevel.sampleSmallContainer(2004);
            raw = c.predictRaw(ds);
            n = ds.numRows();
            % n-by-outputDimension-by-numSamples for a constant-leaf forest.
            testCase.verifySize(raw, [n, 1, c.numSamples()]);
        end

        function containerSplitCountsAreNonNegativeIntegers(testCase)
            [c, ds] = testLowLevel.sampleSmallContainer(2005);
            p = ds.numCovariates();
            counts = c.splitCounts(p);
            testCase.verifyNumElements(counts, p);
            testCase.verifyTrue(all(counts >= 0));
            testCase.verifyEqual(counts, round(counts), ...
                'Split counts must be whole numbers.');

            % Per-draw counts summed over draws should match the pooled total.
            perDraw = zeros(1, p);
            for s = 1:c.numSamples()
                perDraw = perDraw + c.forestSplitCounts(s, p)';
            end
            testCase.verifyEqual(perDraw(:), counts(:), ...
                'Pooled split counts must equal the sum of per-draw counts.');
        end

        %% ---- ForestSampler prior ----------------------------------------

        function samplerPriorRoundTrip(testCase)
            X = rand(30, 3);
            ds = stochtree.Dataset();
            ds.addCovariates(X);
            sampler = stochtree.ForestSampler(ds, zeros(3,1), 10, 30, ...
                0.9, 3.0, 4, 8);

            p = sampler.getPrior();
            testCase.verifyEqual(p.alpha, 0.9, 'AbsTol', 1e-12);
            testCase.verifyEqual(p.beta, 3.0, 'AbsTol', 1e-12);

            % Update the prior and read it back.
            sampler.setPrior(0.5, 1.5, 2, 5);
            p2 = sampler.getPrior();
            testCase.verifyEqual(p2.alpha, 0.5, 'AbsTol', 1e-12);
            testCase.verifyEqual(p2.beta, 1.5, 'AbsTol', 1e-12);
        end

        %% ---- Variance models --------------------------------------------

        function globalVarianceSampleIsPositive(testCase)
            resid = stochtree.Residual(randn(200, 1));
            rngObj = stochtree.RNG(7);
            gv = stochtree.GlobalVarianceModel();
            s = gv.sample(resid, rngObj, 3, 1);
            testCase.verifyGreaterThan(s, 0);
            testCase.verifyTrue(isscalar(s) && isfinite(s));
        end

        function leafVarianceSampleIsPositive(testCase)
            % SampleVarianceParameter reads the forest's leaf/tracker state, so
            % the forest must be driven through the sampler first -- calling it
            % on a bare setRootValue forest is not a supported code path.
            rng(801);
            n = 150; p = 3; numTrees = 20;
            X = rand(n, p);
            resid = X(:,1) + 0.3 * randn(n, 1);

            ds = stochtree.Dataset();
            ds.addCovariates(X);
            residual = stochtree.Residual(resid);
            cppRng = stochtree.RNG(8);
            container = stochtree.ForestContainer(numTrees, 1, true, false);
            forest = stochtree.Forest(numTrees, 1, true, false);
            sampler = stochtree.ForestSampler(ds, zeros(p,1), numTrees, n, ...
                0.95, 2.0, 5, 10);
            sampler.initializeForest(ds, residual, forest, 0, mean(resid));

            opts = struct('cutpointGridSize', 100, 'leafModelScale', var(resid,1)/numTrees, ...
                'variableWeights', repmat(1/p, p, 1), 'aForest', 1, 'bForest', 1, ...
                'globalVariance', var(resid, 1), 'leafModel', 0, ...
                'numFeaturesSubsample', p, 'keepForest', true, 'gfr', true, ...
                'numThreads', 1);
            sampler.sampleOneIteration(container, forest, ds, residual, cppRng, opts);

            lv = stochtree.LeafVarianceModel();
            s = lv.sample(forest, cppRng, 3, 0.5);
            testCase.verifyGreaterThan(s, 0);
            testCase.verifyTrue(isscalar(s) && isfinite(s));
        end

        %% ---- End-to-end manual sampling loop ----------------------------

        function manualBartLoopRecoversSignal(testCase)
            % Reimplement plain BART with the primitives (as demoLowLevel does)
            % and confirm the low-level path fits as well as the high-level one.
            rng(3);
            n = 400; p = 5; numTrees = 50;
            X = rand(n, p);
            y = 10*sin(pi*X(:,1).*X(:,2)) + 10*X(:,4) + randn(n, 1);

            yBar = mean(y); yStd = std(y, 1);
            resid = (y - yBar) / yStd;
            residVar = var(resid, 1);

            ds = stochtree.Dataset();
            ds.addCovariates(X);
            residual = stochtree.Residual(resid);
            cppRng = stochtree.RNG(1234);

            container = stochtree.ForestContainer(numTrees, 1, true, false);
            forest = stochtree.Forest(numTrees, 1, true, false);
            sampler = stochtree.ForestSampler(ds, zeros(p,1), numTrees, n, ...
                0.95, 2.0, 5, 10);
            sampler.initializeForest(ds, residual, forest, 0, mean(resid));

            globalVar = stochtree.GlobalVarianceModel();
            leafVar = stochtree.LeafVarianceModel();

            sigma2 = residVar;
            leafScale = residVar / numTrees;
            opts = struct('cutpointGridSize', 100, 'leafModelScale', leafScale, ...
                'variableWeights', repmat(1/p, p, 1), 'aForest', 1, 'bForest', 1, ...
                'globalVariance', sigma2, 'leafModel', 0, 'numFeaturesSubsample', p, ...
                'keepForest', true, 'gfr', true, 'numThreads', 1);

            numGFR = 10; numMCMC = 100;
            for iter = 1:(numGFR + numMCMC)
                opts.gfr = iter <= numGFR;
                opts.globalVariance = sigma2;
                opts.leafModelScale = leafScale;
                sampler.sampleOneIteration(container, forest, ds, residual, cppRng, opts);
                sigma2 = globalVar.sample(residual, cppRng, 0, 0);
                leafScale = leafVar.sample(forest, cppRng, 3, residVar/numTrees);
            end
            for i = 1:numGFR
                container.deleteSample(1);
            end

            testCase.verifyEqual(container.numSamples(), numMCMC);
            % numLeaves / sumLeafSquared are valid to read on the active forest
            % once it has been driven through the sampler.
            testCase.verifyGreaterThanOrEqual(forest.numLeaves(), numTrees);
            testCase.verifyGreaterThanOrEqual(forest.sumLeafSquared(), 0);
            preds = container.predict(ds) * yStd + yBar;
            yhat = mean(preds, 2);
            r2 = 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);
            testCase.verifyGreaterThan(r2, 0.8, ...
                'The manual BART loop should recover the signal.');
        end
    end

    %% ---- Shared helpers -------------------------------------------------

    methods (Static, Access = private)
        function [container, ds] = sampleSmallContainer(seed)
            %SAMPLESMALLCONTAINER Run a short GFR-only sampler to fill a container.
            rng(seed);
            n = 120; p = 3; numTrees = 15;
            X = rand(n, p);
            resid = X(:,1) + 0.3 * randn(n, 1);

            ds = stochtree.Dataset();
            ds.addCovariates(X);
            residual = stochtree.Residual(resid);
            cppRng = stochtree.RNG(seed);
            container = stochtree.ForestContainer(numTrees, 1, true, false);
            forest = stochtree.Forest(numTrees, 1, true, false);
            sampler = stochtree.ForestSampler(ds, zeros(p,1), numTrees, n, ...
                0.95, 2.0, 5, 10);
            sampler.initializeForest(ds, residual, forest, 0, mean(resid));

            opts = struct('cutpointGridSize', 100, 'leafModelScale', var(resid,1)/numTrees, ...
                'variableWeights', repmat(1/p, p, 1), 'aForest', 1, 'bForest', 1, ...
                'globalVariance', var(resid, 1), 'leafModel', 0, ...
                'numFeaturesSubsample', p, 'keepForest', true, 'gfr', true, ...
                'numThreads', 1);
            for iter = 1:3
                sampler.sampleOneIteration(container, forest, ds, residual, cppRng, opts);
            end
        end

        function deleteIfPresent(f)
            if isfile(f)
                delete(f);
            end
        end
    end
end
