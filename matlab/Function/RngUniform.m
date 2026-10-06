function [U, S] = RngUniform(S, N)
%RNGUNIFORM Uniform numbers in [0, 1) from a portable stream (see RngStream).
%
%   [U, S] = RngUniform(S, N) returns N uniforms (N x 1 double) and the
%   advanced stream. Each step advances all 64 lanes once and emits, lane by
%   lane, the top 53 bits of every state scaled by 2^-53; whole steps are
%   always consumed. The numbers equal those of RandomStream.uniform(N) in
%   the Python version bit for bit.

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
