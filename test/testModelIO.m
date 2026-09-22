classdef testModelIO < matlab.unittest.TestCase
    %TESTMODELIO Tests for model accessor methods and file serialization.
    %
    %   Covers the BARTModel/BCFModel convenience methods that testStochtree.m
    %   does not: posteriorMean, credibleInterval, variableSplitCounts, summary,
    %   chainMatrix error paths, save/load to disk, and the BCF ATE accessors.
    %
    %   The models are fit once in TestClassSetup and reused, since fitting is
    %   the expensive part and these tests only read from the fitted objects.
    %
    %   Run with:  runtests('test/testModelIO.m')

    properties
        BartModel
        BartMultiChain
        BcfModel
        Truth
    end

    methods (TestClassSetup)
        function fitModels(testCase)
            here = fileparts(mfilename('fullpath'));
            toolboxRoot = fileparts(here);
            addpath(toolboxRoot);
            testCase.assertNotEmpty(which('mex_stochtree'), ...
                'mex_stochtree is not on the path. Run build_stochtree first.');

            rng(1001);
            n = 300;
            X = rand(n, 4);
            f = 3 * X(:,1) + 2 * X(:,2).^2;
            y = f + 0.3 * randn(n, 1);
            testCase.Truth = struct('X', X, 'y', y, 'f', f);

            testCase.BartModel = stochtree.bart(X, y, 'NumGFR', 8, 'NumMCMC', 60, ...
                'NumTrees', 40, 'RandomSeed', 1);
            testCase.BartMultiChain = stochtree.bart(X, y, 'NumGFR', 8, ...
                'NumBurnin', 20, 'NumMCMC', 30, 'NumChains', 3, 'NumTrees', 40, ...
                'RandomSeed', 2);

            nb = 400;
            Xb = rand(nb, 4);
            propensity = 0.3 + 0.4 * Xb(:,1);
            Z = double(rand(nb, 1) < propensity);
            tau = 1 + Xb(:,2);
            yb = 2 * Xb(:,1) + tau .* Z + 0.4 * randn(nb, 1);
            testCase.BcfModel = stochtree.bcf(Xb, Z, yb, ...
                'PropensityTrain', propensity, 'NumGFR', 8, 'NumMCMC', 60, ...
                'RandomSeed', 3);
        end
    end

    methods (TestClassTeardown)
        function releaseModels(testCase)
            % Explicitly drop the cached models so their C++ containers are
            % freed before any other test class runs. cleanup_all is avoided
            % here on purpose: it would free the shared models mid-class.
            testCase.BartModel = [];
            testCase.BartMultiChain = [];
            testCase.BcfModel = [];
        end
    end

    %% ---- BART accessor methods -----------------------------------------

    methods (Test)
        function posteriorMeanMatchesManualMean(testCase)
            m = testCase.BartModel;
            X = testCase.Truth.X;
            pm = m.posteriorMean(X);
            manual = mean(m.predict(X, 'samplesOnly', false).yhat, 2);
            testCase.verifyEqual(pm, manual, 'AbsTol', 1e-10);
            testCase.verifySize(pm, [size(X, 1), 1]);
        end

        function credibleIntervalBracketsPosteriorMean(testCase)
            m = testCase.BartModel;
            X = testCase.Truth.X;
            ci = m.credibleInterval(X, 0.9);
            testCase.verifySize(ci, [size(X, 1), 2]);
            % Lower bound below upper bound everywhere.
            testCase.verifyTrue(all(ci(:,1) <= ci(:,2)));
            % The posterior mean should lie inside the interval.
            pm = m.posteriorMean(X);
            testCase.verifyTrue(all(pm >= ci(:,1) - 1e-9));
            testCase.verifyTrue(all(pm <= ci(:,2) + 1e-9));
        end

        function credibleIntervalDefaultLevelIs95(testCase)
            m = testCase.BartModel;
            X = testCase.Truth.X(1:20, :);
            ciDefault = m.credibleInterval(X);
            ci95 = m.credibleInterval(X, 0.95);
            testCase.verifyEqual(ciDefault, ci95, 'AbsTol', 1e-12);
        end

        function credibleIntervalWiderThanNarrowerLevel(testCase)
            m = testCase.BartModel;
            X = testCase.Truth.X(1:30, :);
            wide = m.credibleInterval(X, 0.95);
            narrow = m.credibleInterval(X, 0.5);
            wideWidth = wide(:,2) - wide(:,1);
            narrowWidth = narrow(:,2) - narrow(:,1);
            testCase.verifyTrue(all(wideWidth >= narrowWidth - 1e-9), ...
                'A 95% interval must be at least as wide as a 50% interval.');
        end

        function variableSplitCountsShapeAndTotal(testCase)
            m = testCase.BartModel;
            counts = m.variableSplitCounts();
            testCase.verifyNumElements(counts, m.NumCovariates);
            testCase.verifyTrue(all(counts >= 0));
            % The informative covariates (1 and 2) should attract more splits
            % than the noise covariates on this data.
            testCase.verifyGreaterThan(sum(counts(1:2)), sum(counts(3:4)));
        end

        function summaryReturnsDescriptiveString(testCase)
            m = testCase.BartModel;
            s = m.summary();
            testCase.verifyClass(s, 'char');
            testCase.verifySubstring(s, 'BART');
            testCase.verifySubstring(s, 'draws');
        end

        %% ---- chainMatrix ------------------------------------------------

        function chainMatrixReshapesByChain(testCase)
            m = testCase.BartMultiChain;
            M = m.chainMatrix(m.Sigma2Samples);
            testCase.verifySize(M, [m.NumMCMC, m.NumChains]);
            % Every retained draw must appear exactly once.
            testCase.verifyEqual(sort(M(:)), sort(m.Sigma2Samples(:)), ...
                'AbsTol', 1e-12);
        end

        function chainMatrixRejectsWrongLength(testCase)
            m = testCase.BartMultiChain;
            testCase.verifyError(@() m.chainMatrix(ones(m.NumSamples + 1, 1)), ...
                'stochtree:size');
        end

        function chainMatrixSingleChainReturnsColumn(testCase)
            % With one chain there is nothing to reshape; the vector is returned
            % as-is.
            m = testCase.BartModel;
            M = m.chainMatrix(m.Sigma2Samples);
            testCase.verifyEqual(M, m.Sigma2Samples(:), 'AbsTol', 1e-12);
        end

        %% ---- BART save / load to disk -----------------------------------

        function bartSaveLoadFileRoundTrip(testCase)
            m = testCase.BartModel;
            X = testCase.Truth.X;
            file = [tempname, '.mat'];
            cleanup = onCleanup(@() testModelIO.deleteIfPresent(file)); %#ok<NASGU>

            m.save(file);
            testCase.verifyTrue(isfile(file));

            restored = stochtree.BARTModel.load(file);
            testCase.verifyEqual(restored.NumSamples, m.NumSamples);
            testCase.verifyEqual( ...
                restored.predict(X, 'samplesOnly', false).yhat, ...
                m.predict(X, 'samplesOnly', false).yhat, 'AbsTol', 1e-10, ...
                'save/load must reproduce the model predictions exactly.');
        end

        %% ---- BART predict output conventions ----------------------------

        function predictSamplesOnlyTransposesSubset(testCase)
            % 'samplesOnly' (the default) returns idxSamples-by-n; the struct
            % form returns n-by-NumSamples. They must be transposes for the same
            % index set.
            m = testCase.BartModel;
            X = testCase.Truth.X(1:10, :);
            idx = [1 3 5];
            subset = m.predict(X, 'idxSamples', idx);        % 3-by-10
            full = m.predict(X, 'samplesOnly', false).yhat;  % 10-by-NumSamples
            testCase.verifySize(subset, [numel(idx), size(X, 1)]);
            testCase.verifyEqual(subset, full(:, idx)', 'AbsTol', 1e-12);
        end

        %% ---- BCF accessor methods ---------------------------------------

        function ateSamplesMatchesTauHatMean(testCase)
            m = testCase.BcfModel;
            draws = m.ateSamples();
            testCase.verifyNumElements(draws, m.NumSamples);
            testCase.verifyEqual(draws, mean(m.TauHatTrain, 1)', 'AbsTol', 1e-12);
        end

        function ateSummaryIntervalOrdersAndContainsMean(testCase)
            m = testCase.BcfModel;
            s = m.ateSummary(0.9);
            testCase.verifyEqual(s.level, 0.9);
            testCase.verifyLessThan(s.interval(1), s.interval(2));
            testCase.verifyGreaterThanOrEqual(s.mean, s.interval(1));
            testCase.verifyLessThanOrEqual(s.mean, s.interval(2));
        end

        function posteriorCATEMatchesPredictMean(testCase)
            m = testCase.BcfModel;
            Xnew = rand(40, m.NumCovariates);
            Znew = double(rand(40, 1) < 0.5);
            pi_ = 0.3 + 0.4 * Xnew(:,1);
            cate = m.posteriorCATE(Xnew, Znew, pi_);
            manual = mean(m.predict(Xnew, Znew, pi_).tau, 2);
            testCase.verifyEqual(cate, manual, 'AbsTol', 1e-10);
            testCase.verifySize(cate, [40, 1]);
        end

        function bcfPredictRequiresPropensityWhenSupplied(testCase)
            % This model was fit with an external propensity and no internal
            % model, so prediction must be given propensity scores.
            m = testCase.BcfModel;
            testCase.assertFalse(m.InternalPropensityModel);
            Xnew = rand(10, m.NumCovariates);
            Znew = double(rand(10, 1) < 0.5);
            testCase.verifyError(@() m.predict(Xnew, Znew), 'stochtree:input');
        end

        function bcfSaveLoadFileRoundTrip(testCase)
            m = testCase.BcfModel;
            Xnew = rand(30, m.NumCovariates);
            Znew = double(rand(30, 1) < 0.5);
            pi_ = 0.3 + 0.4 * Xnew(:,1);
            expected = m.predict(Xnew, Znew, pi_).tau;

            file = [tempname, '.mat'];
            cleanup = onCleanup(@() testModelIO.deleteIfPresent(file)); %#ok<NASGU>
            m.save(file);
            restored = stochtree.BCFModel.load(file);
            testCase.verifyEqual(restored.predict(Xnew, Znew, pi_).tau, ...
                expected, 'AbsTol', 1e-10);
        end
    end

    methods (Static, Access = private)
        function deleteIfPresent(f)
            if isfile(f)
                delete(f);
            end
        end
    end
end
