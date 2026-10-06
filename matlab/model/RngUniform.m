function [U, S] = RngUniform(S, N)

if ~isscalar(N) || N < 0 || N ~= fix(N)
    error('NetPRS:RngUniform:N', 'N must be a non-negative integer.');
end
numLane = numel(S.State);
numStep = ceil(N / numLane);
out = zeros(numLane, numStep);
state = S.State;
for step = 1:numStep
    state = RngXorshift64(state);
    out(:, step) = double(bitshift(state, -11)) * 2^-53;
end
S.State = state;
U = out(1:N)';
U = U(:);
end
