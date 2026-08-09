function [R, stats] = rhat(draws, chainIdx)
%STOCHTREE.RHAT Split Gelman-Rubin potential scale reduction factor.
%
%   R = STOCHTREE.RHAT(X) treats X as an iterations-by-chains matrix and
%   returns the split R-hat convergence diagnostic.
%
%   R = STOCHTREE.RHAT(V, CHAINIDX) takes a vector of draws V together with a
%   matching vector of chain labels, as stored in a fitted model's ChainIndex
%   property. Draws labelled 0 (shared warm start draws) are ignored.
%
%   [R, STATS] = ... also returns a struct with the within-chain variance W,
%   the between-chain variance B, the pooled variance estimate, the number of
%   split chains and their length.
%
%   Each chain is split in half before the comparison, which catches chains
%   that are individually drifting but happen to agree with each other.
%
%   At least two chains are required: the diagnostic works by comparing chains
%   against one another, so a single chain is rejected rather than silently
%   reduced to a within-chain comparison of its two halves.
%
%   Values near 1 suggest the chains have mixed. A common rule of thumb is to
%   treat anything above 1.01 as a reason to run longer, though for BART the
%   forest structure itself is not identified, so this is most informative
%   applied to scalar parameters like sigma^2 or to predictions at a point.
%
%   Example:
%       model = stochtree.bart(X, y, 'NumGFR', 10, 'NumChains', 4);
%       stochtree.rhat(model.Sigma2Samples, model.ChainIndex)
%
%   See also STOCHTREE.BART, STOCHTREE.BCF.

if nargin >= 2 && ~isempty(chainIdx)
    draws = draws(:);
    chainIdx = chainIdx(:);
    if numel(draws) ~= numel(chainIdx)
        error('stochtree:size', 'draws and chainIdx must have the same length.');
    end
    labels = unique(chainIdx(chainIdx > 0));
    if isempty(labels)
        error('stochtree:value', 'chainIdx contains no chain labels above 0.');
    end
    if numel(labels) < 2
        error('stochtree:input', ...
            ['R-hat compares chains against each other, and chainIdx contains ' ...
             'only %d chain. Refit with ''NumChains'', 4 (or more) to get a ' ...
             'convergence diagnostic.'], numel(labels));
    end
    counts = arrayfun(@(c) sum(chainIdx == c), labels);
    if numel(unique(counts)) ~= 1
        error('stochtree:value', ...
            'Chains have unequal lengths (%s); R-hat needs equal-length chains.', ...
            mat2str(counts(:)'));
    end
    X = zeros(counts(1), numel(labels));
    for c = 1:numel(labels)
        X(:, c) = draws(chainIdx == labels(c));
    end
else
    X = draws;
    if isvector(X)
        error('stochtree:input', ...
            ['R-hat compares chains against each other, and a vector is a ' ...
             'single chain. Pass an iterations-by-chains matrix, or supply a ' ...
             'chain index vector as the second argument.']);
    end
end

[n, m] = size(X);
if n < 4
    error('stochtree:value', 'Need at least 4 draws per chain to split them.');
end

% Split each chain in half, discarding a middle draw when n is odd.
half = floor(n / 2);
S = [X(1:half, :), X(n - half + 1:end, :)];
[nSplit, mSplit] = size(S);

chainMeans = mean(S, 1);
chainVars = var(S, 0, 1);                 % unbiased, denominator nSplit-1

W = mean(chainVars);
B = nSplit * var(chainMeans, 0, 2);       % between-chain variance
varPlus = ((nSplit - 1) / nSplit) * W + B / nSplit;

if W <= 0
    R = NaN;
else
    R = sqrt(varPlus / W);
end

if nargout > 1
    stats.W = W;
    stats.B = B;
    stats.varPlus = varPlus;
    stats.numSplitChains = mSplit;
    stats.splitChainLength = nSplit;
    stats.chainMeans = chainMeans;
end
end
