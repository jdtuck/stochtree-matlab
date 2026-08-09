function M = asMatrix(A, name)
%ASMATRIX Validate and coerce input to a real 2-D double matrix.

if nargin < 2, name = 'input'; end
if istable(A)
    A = table2array(A);
end
if ~isnumeric(A) && ~islogical(A)
    error('stochtree:type', '%s must be numeric or logical.', name);
end
if ~ismatrix(A)
    error('stochtree:type', '%s must be a 2-D matrix.', name);
end
M = double(A);
if ~isreal(M)
    error('stochtree:type', '%s must be real-valued.', name);
end
end
