function [X, Mean] = ImputeGenotype(X, RefCols)

if nargin < 2 || isempty(RefCols)
    RefCols = 1:size(X, 2);
end
X = double(X);
R = X(:, RefCols);
obs = isfinite(R);
R(~obs) = 0;
cnt = sum(obs, 2);
Mean = sum(R, 2) ./ max(cnt, 1);
if any(cnt == 0 & any(~isfinite(X), 2))
    error('NetPRS:ImputeGenotype:NoReference', 'A SNP with missing values has no genotyped reference subject.');
end
[ii, jj] = find(~isfinite(X));
X(sub2ind(size(X), ii, jj)) = Mean(ii);
end
