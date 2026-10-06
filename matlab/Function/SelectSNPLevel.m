function [Idx, Info] = SelectSNPLevel(P, Thresholds)
%SELECTSNPLEVEL SNP screening into significance levels (Sec. IV-B, Table III).
%
%   [Idx, Info] = SelectSNPLevel(P, Thresholds)
%
%   SNP j belongs to level l when -log10(P_j) > Thresholds(l). The paper
%   used Thresholds = [3 4 5] for ADNI and [4 5 6] for BICWALZS
%   (974/130/15 and 420/63/13 SNPs, respectively).
%
%   Inputs
%     P           [s x 1] GWAS P-values (NaN = not tested, never selected)
%     Thresholds  [1 x l] -log10(P) cut-offs
%   Outputs
%     Idx   {1 x l} SNP indices of each level, in the original SNP order
%     Info  .NumSNP [1 x l], .Thresholds, .Fraction (share of the tested SNPs)

P = double(P(:));
if any(P(isfinite(P)) < 0 | P(isfinite(P)) > 1)
    error('NetPRS:SelectSNPLevel:InvalidP', 'P-values must lie in [0, 1].');
end
score = -log10(P);
numLevel = numel(Thresholds);
Idx = cell(1, numLevel);
numSNP = zeros(1, numLevel);
for l = 1:numLevel
    Idx{l} = find(score > Thresholds(l));
    numSNP(l) = numel(Idx{l});
end
Info.NumSNP = numSNP;
Info.Thresholds = Thresholds(:)';
Info.Fraction = numSNP / sum(isfinite(P));
end
