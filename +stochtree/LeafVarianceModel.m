classdef LeafVarianceModel < handle
    %STOCHTREE.LEAFVARIANCEMODEL Gibbs sampler for the leaf node variance.
    %
    %   Draws tau (the prior variance of a constant leaf) from its inverse-gamma
    %   full conditional given the current forest's leaf values. Only defined for
    %   constant-leaf forests.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = LeafVarianceModel()
            obj.Handle = mex_stochtree('leaf_var_create');
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function tau = sample(obj, forest, rng, a, b)
            %SAMPLE Draw one value of the leaf scale given IG(a, b) hyperparameters.
            tau = mex_stochtree('leaf_var_sample', obj.Handle, forest.Handle, ...
                rng.Handle, double(a), double(b));
        end
    end
end
