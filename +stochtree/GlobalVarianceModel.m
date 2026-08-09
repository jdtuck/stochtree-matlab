classdef GlobalVarianceModel < handle
    %STOCHTREE.GLOBALVARIANCEMODEL Gibbs sampler for the global error variance.
    %
    %   Draws sigma^2 from its inverse-gamma full conditional, which depends on
    %   the rest of the model only through the full residual.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = GlobalVarianceModel()
            obj.Handle = mex_stochtree('global_var_create');
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function sigma2 = sample(obj, residual, rng, a, b)
            %SAMPLE Draw one value of sigma^2 given IG(a, b) prior hyperparameters.
            sigma2 = mex_stochtree('global_var_sample', obj.Handle, residual.Handle, ...
                rng.Handle, double(a), double(b));
        end
    end
end
