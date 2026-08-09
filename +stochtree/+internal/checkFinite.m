function checkFinite(A, name)
%CHECKFINITE Reject NaN/Inf, which the C++ samplers cannot handle in outcomes.

if any(~isfinite(A(:)))
    error('stochtree:value', ...
        '%s contains NaN or Inf values. Remove or impute them before sampling.', name);
end
end
