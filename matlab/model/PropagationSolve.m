function Y = PropagationSolve(Net, B)

if ~isempty(Net.D)
    Y = Net.V * (Net.D .* (Net.V' * B));
elseif ~isempty(Net.R)
    Y = Net.R \ (Net.R' \ B);
else
    error('NetPRS:PropagationSolve:NotPrepared', ...
        'Propagation operator of network %s is not prepared; call ParamReshape first.', Net.Name);
end
end
