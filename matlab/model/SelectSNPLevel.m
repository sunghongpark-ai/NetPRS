function [Idx, Info] = SelectSNPLevel(P, Thresholds)

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
