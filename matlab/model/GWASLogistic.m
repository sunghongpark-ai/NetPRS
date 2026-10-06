function Res = GWASLogistic(X, Y, Cov, Opts)

if nargin < 3
    Cov = [];
end
if nargin < 4 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'ChunkSize', 5000);

[s, n] = size(X);
Y = double(Y(:));
if numel(Y) ~= n
    error('NetPRS:GWASLogistic:SizeMismatch', 'Y must have one entry per column of X.');
end
if isempty(Cov)
    Cov = zeros(n, 0);
end
if size(Cov, 1) ~= n
    error('NetPRS:GWASLogistic:SizeMismatch', 'Cov must have n rows.');
end
C = [ones(n, 1), double(Cov)];

Res.Beta = nan(s, 1);
Res.SE = nan(s, 1);
Res.NumObs = zeros(s, 1);
Res.Converged = false(s, 1);
for first = 1:Opts.ChunkSize:s
    idx = first:min(first + Opts.ChunkSize - 1, s);
    V = double(X(idx, :))';
    M = isfinite(V);
    Out = LogisticBatch(Y, C, reshape(V, n, numel(idx), 1), M, Opts);
    Res.Beta(idx) = Out.Coef(end, :)';
    Res.SE(idx) = Out.SE(end, :)';
    Res.NumObs(idx) = Out.NumObs';
    Res.Converged(idx) = Out.Converged';
end
Res.OR = exp(Res.Beta);
Res.Stat = Res.Beta ./ Res.SE;
Res.P = erfc(abs(Res.Stat) / sqrt(2));
end
