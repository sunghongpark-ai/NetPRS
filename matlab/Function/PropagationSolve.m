function Y = PropagationSolve(Net, B)
%PROPAGATIONSOLVE Network propagation Y = (I + mu*L)^{-1} * B (Eq. (3)/(4)).
%
%   Y = PropagationSolve(Net, B) applies the propagation operator prepared
%   by ParamReshape for the current smoothness mu of the network Net:
%     eig  representation : Y = V * (D .* (V' * B)),  D = 1./(1 + mu*lambda)
%     chol representation : Y = R \ (R' \ B),         R'R = I + mu*L
%   Both evaluate the same closed-form GSSL solution; 'eig' needs one
%   eigendecomposition per network and O(s^2) work per right-hand side,
%   'chol' re-factorizes I + mu*L whenever mu changes.
%
%   Inputs
%     Net  element of model.Net (fields V, D or R)
%     B    [s x c] right-hand sides
%   Output
%     Y    [s x c]

if ~isempty(Net.D)
    Y = Net.V * (Net.D .* (Net.V' * B));
elseif ~isempty(Net.R)
    Y = Net.R \ (Net.R' \ B);
else
    error('NetPRS:PropagationSolve:NotPrepared', ...
        'Propagation operator of network %s is not prepared; call ParamReshape first.', Net.Name);
end
end
