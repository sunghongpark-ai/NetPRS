function [P, S] = RngPermutation(S, N)

[U, S] = RngUniform(S, N);
[~, P] = sort(U);
P = P(:);
end
