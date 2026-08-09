classdef RNG < handle
    %STOCHTREE.RNG C++ random number generator used by the samplers.
    %
    %   stochtree.RNG(seed) seeds a std::mt19937. Pass a negative seed (the
    %   default) to seed non-deterministically from std::random_device.
    %
    %   Note that this generator is separate from MATLAB's own RNG. Reproducing
    %   a run needs both seeded, since stochtree.bcf also draws from MATLAB's
    %   generator for the adaptive coding parameters.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = RNG(seed)
            if nargin < 1 || isempty(seed), seed = -1; end
            obj.Handle = mex_stochtree('rng_create', double(seed));
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end
    end
end
