function S = NetworkSummary(Wp, Wg, Label)
%NETWORKSUMMARY Statistics of the SNP interaction networks (Table III).
%
%   S = NetworkSummary(Wp, Wg)
%   S = NetworkSummary(Wp, Wg, Label)   also prints one Table-III-like row
%
%   For each network (undirected, edges = non-zero upper-triangle entries):
%     NumEdge, AvgEdge (mean edge weight), Density = NumEdge / (s(s-1)/2).
%   For the edges present in both networks ("common SNP-SNP interactions"):
%     NumCommon, Corr (Pearson correlation of the two edge weights) and the
%     two-sample Kolmogorov-Smirnov P-value comparing the two weight
%     distributions. Both are NaN ('-' in Table III) with fewer than 3
%     common edges, where the correlation is trivially +/-1.
%
%   Inputs
%     Wp, Wg  [s x s] phenotypic and genomic networks
%     Label   optional text printed in front of the row
%   Output
%     S  struct with fields NumNode, Phe, Gen (NumEdge, AvgEdge, Density),
%        NumCommon, Corr, KSP

s = size(Wp, 1);
if ~isequal(size(Wp), [s, s]) || ~isequal(size(Wg), [s, s])
    error('NetPRS:NetworkSummary:SizeMismatch', 'Wp and Wg must be s x s matrices.');
end
upperMask = triu(true(s), 1);
wp = full(Wp(upperMask));
wg = full(Wg(upperMask));
numPair = s * (s - 1) / 2;
S.NumNode = s;
S.Phe = Describe(wp, numPair);
S.Gen = Describe(wg, numPair);
common = (wp ~= 0) & (wg ~= 0);
S.NumCommon = nnz(common);
if S.NumCommon >= 3
    c = corrcoef(wp(common), wg(common));
    S.Corr = c(1, 2);
    S.KSP = KSTest2Asymptotic(wp(common), wg(common));
else
    S.Corr = NaN;
    S.KSP = NaN;
end
if nargin >= 3 && ~isempty(Label)
    fprintf('%-28s %6d | %8d %7.4f %8.4f%% | %8d %7.4f %8.4f%% | %6d %8.4f %10.2e\n', ...
        Label, s, S.Phe.NumEdge, S.Phe.AvgEdge, 100 * S.Phe.Density, ...
        S.Gen.NumEdge, S.Gen.AvgEdge, 100 * S.Gen.Density, S.NumCommon, S.Corr, S.KSP);
end
end

function D = Describe(w, numPair)
e = w ~= 0;
D.NumEdge = nnz(e);
if D.NumEdge > 0
    D.AvgEdge = mean(w(e));
else
    D.AvgEdge = NaN;
end
D.Density = D.NumEdge / numPair;
end
