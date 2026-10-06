function [X, Mean] = ImputeGenotype(X, RefCols)
%IMPUTEGENOTYPE Mean imputation of missing genotypes (declared preprocessing).
%
%   [X, Mean] = ImputeGenotype(X)
%   [X, Mean] = ImputeGenotype(X, RefCols)
%
%   NaN entries of SNP j are replaced by the mean dosage of SNP j over the
%   reference subjects RefCols (default: all subjects). Use the discovery
%   cohort as reference so that the validation cohort does not inform the
%   imputation. The paper does not describe missing-genotype handling; the
%   NetPRS model itself requires complete genotypes (call rate > 0.99 after
%   quality control leaves few missing values).
%
%   Inputs
%     X        [s x n] dosage with NaN for missing genotypes
%     RefCols  index or logical vector of reference subjects
%   Outputs
%     X        [s x n] imputed dosage
%     Mean     [s x 1] imputation values

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
