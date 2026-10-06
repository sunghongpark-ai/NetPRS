function S = EffectSignificance(X, Z, Y, Type)
%EFFECTSIGNIFICANCE Significance of independent vs. integrated SNP effects.
%
%   S = EffectSignificance(X, Z, Y)
%   S = EffectSignificance(X, Z, Y, Type)
%
%   Reproduces the statistical analysis of Fig. 4(a): for every SNP the
%   AD-positive and AD-negative groups are compared with a two-sample t-test
%   on the independent effect X (P_X) and on the integrated effect Z (P_Z).
%   The test is not named in the paper; the pooled-variance t-test (the
%   MATLAB ttest2 default) is used unless Type = 'welch'.
%
%   Inputs
%     X, Z  [s x n] independent effect and integrated effect
%           (Z from EffectExtraction for the same subjects)
%     Y     [1 x n] diagnosis (1 = AD, 0 = non-AD)
%     Type  'pooled' (default) | 'welch'
%
%   Output
%     S.PX, S.PZ        [s x 1] P-values
%     S.ZMoreSignificant [s x 1] logical, P_Z < P_X; P-values equal within a
%                       relative 1e-9 (e.g. Z = X when every mu is 0) are a
%                       tie and are not counted, so rounding cannot decide
%     S.NumZMoreSignificant, S.FracZMoreSignificant
%     S.MeanLogPX, S.MeanLogPZ   mean of -log10 P over SNPs
%     S.RelativeGain    MeanLogPZ / MeanLogPX - 1 (19.8% / 49.5% for ADNI /
%                       BICWALZS level 2 in Fig. 4(a))

if nargin < 4 || isempty(Type)
    Type = 'pooled';
end
Y = logical(Y(:)');
if size(X, 2) ~= numel(Y) || ~isequal(size(X), size(Z))
    error('NetPRS:EffectSignificance:SizeMismatch', 'X and Z must be s x n with n = numel(Y).');
end
S.PX = TwoSampleTTest(X(:, Y), X(:, ~Y), Type);
S.PZ = TwoSampleTTest(Z(:, Y), Z(:, ~Y), Type);
S.ZMoreSignificant = S.PZ < S.PX * (1 - 1e-9);
S.NumZMoreSignificant = nnz(S.ZMoreSignificant);
S.FracZMoreSignificant = S.NumZMoreSignificant / numel(S.PX);
lx = -log10(S.PX);
lz = -log10(S.PZ);
ok = isfinite(lx) & isfinite(lz);
S.MeanLogPX = mean(lx(ok));
S.MeanLogPZ = mean(lz(ok));
S.RelativeGain = S.MeanLogPZ / S.MeanLogPX - 1;
S.Type = Type;
end
