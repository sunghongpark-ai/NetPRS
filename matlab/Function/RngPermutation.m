function [P, S] = RngPermutation(S, N)
%RNGPERMUTATION Random permutation of 1..N from a portable stream.
%
%   [P, S] = RngPermutation(S, N) returns P [N x 1], the stable sort order of
%   N uniforms drawn with RngUniform. It equals RandomStream.permutation(N)
%   of the Python version plus one (1-based indices).

[U, S] = RngUniform(S, N);
[~, P] = sort(U);
P = P(:);
end
