function releaseHandle(handle)
%RELEASEHANDLE Free a C++ object, tolerating an already-unloaded MEX file.
%
%   Called from the destructors of the stochtree handle classes. During MATLAB
%   shutdown, or after "clear mex", the gateway may already be gone; freeing is
%   then a no-op because the C++ registry went with it.

if isempty(handle) || handle == 0
    return
end
try
    mex_stochtree('delete', handle);
catch
    % MEX file unloaded or never built; nothing left to free.
end
end
