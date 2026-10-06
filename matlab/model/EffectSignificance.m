function S = EffectSignificance(X, Z, Y, Type)

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
