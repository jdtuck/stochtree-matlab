function build_stochtree(varargin)
%BUILD_STOCHTREE Compile the stochtree MEX gateway.
%
%   BUILD_STOCHTREE() compiles the gateway, looking for a stochtree checkout in
%   the usual places (a 'stochtree' folder next to or inside this one).
%
%   BUILD_STOCHTREE('StochtreeRoot', PATH) points the build at a specific
%   stochtree checkout. The checkout must have been cloned recursively:
%
%       git clone --recursive https://github.com/StochasticTree/stochtree.git
%
%   If you already cloned without --recursive, run
%   'git submodule update --init --recursive' inside the checkout first.
%
%   Options
%     'StochtreeRoot'  Path to the stochtree repository.
%     'OpenMP'         Enable OpenMP for multithreaded sampling. Default false.
%                      Requires a compiler with OpenMP support; note that Apple
%                      Clang does not ship with it.
%     'Debug'          Compile with debug symbols and no optimization.
%     'Verbose'        Show the full compiler command line.
%     'AddToPath'      Add the toolbox folder to the MATLAB path and save it.
%                      Default true.
%
%   Example:
%       build_stochtree('StochtreeRoot', '~/src/stochtree', 'OpenMP', true);
%
%   See also STOCHTREE.BART, STOCHTREE.BCF.

q = inputParser;
q.FunctionName = 'build_stochtree';
addParameter(q, 'StochtreeRoot', '');
if ismac
    addParameter(q, 'OpenMP', false);
else
    addParameter(q, 'OpenMP', true);
end
addParameter(q, 'Debug', false);
addParameter(q, 'Verbose', false);
addParameter(q, 'AddToPath', true);
parse(q, varargin{:});
opts = q.Results;

toolboxRoot = fileparts(mfilename('fullpath'));
srcFile = fullfile(toolboxRoot, 'src', 'mex_stochtree.cpp');
if ~isfile(srcFile)
    error('stochtree:build', 'Cannot find the gateway source at %s.', srcFile);
end

stRoot = iResolveStochtreeRoot(opts.StochtreeRoot, toolboxRoot);
fprintf('Using stochtree checkout: %s\n', stRoot);

%% ---- Locate headers ---------------------------------------------------
includeDirs = {fullfile(stRoot, 'include')};
depSpecs = { ...
    'Boost.Math',          fullfile(stRoot, 'deps', 'boost_math', 'include'), 'boost'; ...
    'Eigen',               fullfile(stRoot, 'deps', 'eigen'),                 'Eigen'; ...
    'fast_double_parser',  fullfile(stRoot, 'deps', 'fast_double_parser', 'include'), 'fast_double_parser.h'; ...
    'fmt',                 fullfile(stRoot, 'deps', 'fmt', 'include'),        'fmt'};

missing = {};
for i = 1:size(depSpecs, 1)
    dir_i = depSpecs{i, 2};
    probe = fullfile(dir_i, depSpecs{i, 3});
    if isfolder(dir_i) && (isfolder(probe) || isfile(probe))
        includeDirs{end+1} = dir_i; %#ok<AGROW>
    else
        missing{end+1} = depSpecs{i, 1}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('stochtree:build', ...
        ['Missing bundled dependencies: %s.\nThese are git submodules. Run the ' ...
         'following inside %s:\n    git submodule update --init --recursive'], ...
        strjoin(missing, ', '), stRoot);
end

%% ---- Core sources -----------------------------------------------------
% Mirrors the SOURCES list in stochtree's CMakeLists.txt, restricted to files
% that actually exist (the CMake list carries a stale json11.cpp entry).
coreNames = {'container', 'cutpoint_candidates', 'data', 'io', 'leaf_model', ...
    'ordinal_sampler', 'partition_tracker', 'random_effects', 'tree'};
coreSources = {};
for i = 1:numel(coreNames)
    f = fullfile(stRoot, 'src', [coreNames{i} '.cpp']);
    if isfile(f)
        coreSources{end+1} = f; %#ok<AGROW>
    end
end
if isempty(coreSources)
    error('stochtree:build', 'No stochtree core sources found under %s/src.', stRoot);
end

%% ---- Compiler flags ---------------------------------------------------
isMSVC = ispc && ~isempty(strfind(lower(iCompilerName()), 'microsoft')); %#ok<STREMP>

if isMSVC
    cxxFlags = '/std:c++17 /EHsc /bigobj';
    if opts.OpenMP, cxxFlags = [cxxFlags ' /openmp']; end
    flagArgs = {['COMPFLAGS=$COMPFLAGS ' cxxFlags]};
else
    cxxFlags = '-std=c++17 -fPIC';
    if opts.OpenMP, cxxFlags = [cxxFlags ' -fopenmp']; end
    flagArgs = {['CXXFLAGS=$CXXFLAGS ' cxxFlags]};
    if opts.OpenMP
        flagArgs{end+1} = 'LDFLAGS=$LDFLAGS -fopenmp';
    end
end

args = {'-R2018a', '-outdir', toolboxRoot, '-output', 'mex_stochtree'};
if opts.Debug
    args{end+1} = '-g';
else
    args{end+1} = '-O';
end
if opts.Verbose
    args{end+1} = '-v';
end
for i = 1:numel(includeDirs)
    args{end+1} = ['-I' includeDirs{i}]; %#ok<AGROW>
end
args = [args, flagArgs];

allSources = [{srcFile}, coreSources];

fprintf('Compiling %d source files. This usually takes a couple of minutes.\n', ...
    numel(allSources));
try
    mex(args{:}, allSources{:});
catch err
    fprintf(2, '\nThe MEX build failed.\n');
    fprintf(2, 'Check that a C++17 compiler is configured: run "mex -setup C++".\n');
    if opts.OpenMP
        fprintf(2, 'Try rebuilding without OpenMP if the error mentions omp.h.\n');
    end
    rethrow(err);
end

built = fullfile(toolboxRoot, ['mex_stochtree.' mexext]);
fprintf('Built %s\n', built);

if opts.AddToPath
    addpath(toolboxRoot);
    fprintf('Added %s to the MATLAB path.\n', toolboxRoot);
    fprintf('Run "savepath" to make this permanent.\n');
end

% Smoke test: create and free one C++ object.
try
    h = mex_stochtree('dataset_create');
    mex_stochtree('delete', h);
    fprintf('Smoke test passed. Try: runtests(fullfile(''%s'', ''test''))\n', toolboxRoot);
catch err
    warning('stochtree:build', 'Built, but the smoke test failed: %s', err.message);
end
end

% -------------------------------------------------------------------------
function root = iResolveStochtreeRoot(userPath, toolboxRoot)
if ~isempty(userPath)
    root = char(userPath);
    if ~isfolder(root)
        error('stochtree:build', 'StochtreeRoot "%s" is not a folder.', root);
    end
    if ~isfolder(fullfile(root, 'include', 'stochtree'))
        error('stochtree:build', ...
            '"%s" does not look like a stochtree checkout (no include/stochtree).', root);
    end
    return
end

candidates = { ...
    fullfile(toolboxRoot, 'stochtree'), ...
    fullfile(fileparts(toolboxRoot), 'stochtree'), ...
    fullfile(toolboxRoot, '..', 'stochtree')};
for i = 1:numel(candidates)
    if isfolder(fullfile(candidates{i}, 'include', 'stochtree'))
        root = iCanonical(candidates{i});
        return
    end
end
error('stochtree:build', ...
    ['Could not find a stochtree checkout. Clone it with\n' ...
     '    git clone --recursive https://github.com/StochasticTree/stochtree.git\n' ...
     'and then call build_stochtree(''StochtreeRoot'', ''/path/to/stochtree'').']);
end

function p = iCanonical(p)
d = dir(p);
if ~isempty(d)
    p = d(1).folder;
end
end

function name = iCompilerName()
name = '';
try
    cfg = mex.getCompilerConfigurations('C++', 'Selected');
    if ~isempty(cfg)
        name = cfg(1).Name;
    end
catch
    % No compiler configured yet; the mex call below will report it.
end
end
