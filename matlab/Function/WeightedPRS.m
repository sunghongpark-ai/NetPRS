function Score = WeightedPRS(X, Weight)
%WEIGHTEDPRS Weighted polygenic risk score (wPRS baseline, Choi et al., 2020).
%
%   Score = WeightedPRS(X, Weight)
%
%   PRS_i = sum_j Weight_j * X(j, i): the sum of risk-allele dosages weighted
%   by their GWAS effect sizes (log odds ratios estimated in the discovery
%   cohort). This is the independent-effect baseline "wPRS" of Fig. 3(a).
%
%   Inputs
%     X       [s x n] allele dosage (complete; impute missing values first)
%     Weight  [s x 1] effect sizes (e.g. GWASLogistic(...).Beta); NaN
%             weights (failed GWAS fits) contribute 0
%   Output
%     Score   [1 x n]

Weight = double(Weight(:));
if size(X, 1) ~= numel(Weight)
    error('NetPRS:WeightedPRS:SizeMismatch', 'X must have numel(Weight) rows.');
end
if any(~isfinite(X(:)))
    error('NetPRS:WeightedPRS:MissingGenotype', 'X contains NaN/Inf; impute missing genotypes first.');
end
Weight(~isfinite(Weight)) = 0;
Score = Weight' * double(X);
end
