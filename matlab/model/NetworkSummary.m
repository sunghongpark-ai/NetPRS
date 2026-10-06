function S = NetworkSummary(Wp, Wg, Label)

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
