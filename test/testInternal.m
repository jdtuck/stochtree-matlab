classdef testInternal < matlab.unittest.TestCase
    %TESTINTERNAL Tests for pure-MATLAB helpers and rhat edge cases.
    %
    %   These exercise the input-validation layer and the R-hat computation
    %   directly. None of them touch the MEX gateway, so they run fast and
    %   isolate the plain-MATLAB logic from the C++ sampler.
    %
    %   Run with:  runtests('test/testInternal.m')

    methods (TestClassSetup)
        function addToolboxToPath(~)
            here = fileparts(mfilename('fullpath'));
            toolboxRoot = fileparts(here);
            addpath(toolboxRoot);
        end
    end

    %% ---- asMatrix -------------------------------------------------------

    methods (Test)
        function asMatrixPassesThroughDouble(testCase)
            X = rand(5, 3);
            testCase.verifyEqual(stochtree.internal.asMatrix(X, 'X'), X);
        end

        function asMatrixCastsLogicalAndInteger(testCase)
            % Logical and integer inputs are legitimate covariates; they must
            % come back as doubles rather than being rejected.
            L = logical([1 0; 0 1]);
            testCase.verifyEqual(stochtree.internal.asMatrix(L, 'X'), double(L));
            testCase.verifyClass(stochtree.internal.asMatrix(L, 'X'), 'double');

            I = int32([1 2; 3 4]);
            testCase.verifyEqual(stochtree.internal.asMatrix(I, 'X'), double(I));
            testCase.verifyClass(stochtree.internal.asMatrix(I, 'X'), 'double');
        end

        function asMatrixCoercesTable(testCase)
            T = table((1:3)', (4:6)');
            testCase.verifyEqual(stochtree.internal.asMatrix(T, 'X'), ...
                [(1:3)', (4:6)']);
        end

        function asMatrixRejectsComplex(testCase)
            testCase.verifyError(@() stochtree.internal.asMatrix(1 + 2i, 'X'), ...
                'stochtree:type');
        end

        function asMatrixRejectsNonNumeric(testCase)
            testCase.verifyError(@() stochtree.internal.asMatrix('abc', 'X'), ...
                'stochtree:type');
            testCase.verifyError(@() stochtree.internal.asMatrix({1, 2}, 'X'), ...
                'stochtree:type');
        end

        function asMatrixRejectsHigherDimensional(testCase)
            testCase.verifyError(@() stochtree.internal.asMatrix(rand(2,2,2), 'X'), ...
                'stochtree:type');
        end

        function asMatrixDefaultNameDoesNotError(testCase)
            % Called with a single argument the helper still validates.
            testCase.verifyEqual(stochtree.internal.asMatrix([1 2; 3 4]), ...
                [1 2; 3 4]);
        end

        %% ---- checkFinite ------------------------------------------------

        function checkFiniteAcceptsFiniteValues(testCase)
            testCase.verifyWarningFree(@() ...
                stochtree.internal.checkFinite([1 -2 3.5 0], 'y'));
        end

        function checkFiniteRejectsNaN(testCase)
            testCase.verifyError(@() stochtree.internal.checkFinite([1 NaN 3], 'y'), ...
                'stochtree:value');
        end

        function checkFiniteRejectsInf(testCase)
            testCase.verifyError(@() stochtree.internal.checkFinite([1 Inf 3], 'y'), ...
                'stochtree:value');
            testCase.verifyError(@() stochtree.internal.checkFinite([-Inf 2], 'y'), ...
                'stochtree:value');
        end

        %% ---- releaseHandle ----------------------------------------------

        function releaseHandleToleratesZero(testCase)
            % A zero or empty handle means "nothing was ever allocated"; the
            % destructor helper must treat it as a no-op, not an error.
            testCase.verifyWarningFree(@() stochtree.internal.releaseHandle(uint64(0)));
            testCase.verifyWarningFree(@() stochtree.internal.releaseHandle([]));
        end

        %% ---- rhat (matrix form) -----------------------------------------

        function rhatIdenticalDistributionChainsNearOne(testCase)
            % Many independent chains drawn from the same distribution should
            % give R-hat close to 1: within- and between-chain variance agree.
            rng(41);
            X = randn(2000, 8);
            R = stochtree.rhat(X);
            testCase.verifyGreaterThan(R, 0.97);
            testCase.verifyLessThan(R, 1.05);
        end

        function rhatIsPermutationInvariant(testCase)
            % R-hat compares chains symmetrically, so reordering the columns
            % must not change the value.
            rng(42);
            X = randn(50, 4) + [0 1 2 3];
            R = stochtree.rhat(X);
            Rperm = stochtree.rhat(X(:, [3 1 4 2]));
            testCase.verifyEqual(Rperm, R, 'AbsTol', 1e-12);
        end

        function rhatMatrixReturnsStats(testCase)
            rng(1);
            X = randn(40, 3);
            [R, stats] = stochtree.rhat(X);
            testCase.verifyGreaterThan(R, 0);
            testCase.verifyEqual(stats.numSplitChains, 6);   % 3 chains x 2 halves
            testCase.verifyEqual(stats.splitChainLength, 20);
            testCase.verifyEqual(numel(stats.chainMeans), 6);
            testCase.verifyGreaterThanOrEqual(stats.W, 0);
            testCase.verifyGreaterThanOrEqual(stats.B, 0);
        end

        function rhatRejectsTooFewDraws(testCase)
            % Fewer than 4 draws per chain cannot be split into halves.
            testCase.verifyError(@() stochtree.rhat(randn(3, 4)), 'stochtree:value');
        end

        function rhatVectorIsRejected(testCase)
            % A bare vector is a single chain with no index; must be rejected.
            testCase.verifyError(@() stochtree.rhat(randn(50, 1)), 'stochtree:input');
        end

        function rhatIndexLengthMismatchRejected(testCase)
            testCase.verifyError( ...
                @() stochtree.rhat(randn(10, 1), ones(9, 1)), 'stochtree:size');
        end

        function rhatIndexAllZeroRejected(testCase)
            % All draws labelled 0 (warm start only) leaves no chains to compare.
            testCase.verifyError( ...
                @() stochtree.rhat(randn(10, 1), zeros(10, 1)), 'stochtree:value');
        end

        function rhatIndexUnequalChainsRejected(testCase)
            draws = randn(15, 1);
            idx = [ones(10, 1); 2 * ones(5, 1)];
            testCase.verifyError(@() stochtree.rhat(draws, idx), 'stochtree:value');
        end

        function rhatIndexAndMatrixAgree(testCase)
            % The two calling conventions must give the same number on the same
            % data.
            rng(2);
            c1 = randn(30, 1);
            c2 = randn(30, 1) + 0.5;
            fromMatrix = stochtree.rhat([c1, c2]);
            fromIndex = stochtree.rhat([c1; c2], [ones(30,1); 2*ones(30,1)]);
            testCase.verifyEqual(fromIndex, fromMatrix, 'AbsTol', 1e-12);
        end

        function rhatSeparatedChainsExceedOne(testCase)
            X = randn(100, 4) + [0 4 8 12];
            testCase.verifyGreaterThan(stochtree.rhat(X), 1.5);
        end
    end
end
