classdef Residual < handle
    %STOCHTREE.RESIDUAL Continuously updated partial residual.
    %
    %   Wraps StochTree::ColumnVector. During sampling this object holds the
    %   outcome net of every model term that has already been drawn, and the
    %   samplers mutate it in place.
    %
    %   See also STOCHTREE.DATASET, STOCHTREE.FORESTSAMPLER.

    properties (SetAccess = private)
        Handle uint64 = uint64(0)
    end

    methods
        function obj = Residual(y)
            obj.Handle = mex_stochtree('residual_create', double(y(:)));
        end

        function delete(obj)
            stochtree.internal.releaseHandle(obj.Handle);
        end

        function v = getValues(obj)
            v = mex_stochtree('residual_get', obj.Handle);
        end

        function replace(obj, v)
            %REPLACE Overwrite the residual with new values.
            mex_stochtree('residual_replace', obj.Handle, double(v(:)));
        end

        function add(obj, v)
            %ADD Add a vector to the residual, elementwise.
            mex_stochtree('residual_add', obj.Handle, double(v(:)));
        end

        function subtract(obj, v)
            %SUBTRACT Subtract a vector from the residual, elementwise.
            mex_stochtree('residual_subtract', obj.Handle, double(v(:)));
        end
    end
end
