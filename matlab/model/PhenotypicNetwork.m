function [Wp, Info] = PhenotypicNetwork(Arg1, Arg2, Arg3)

if isstruct(Arg1)
    ORint = Arg1.ORint;
    Pval = Arg1.P;
    if nargin >= 2
        Opts = Arg2;
    else
        Opts = struct();
    end
else
    if nargin < 2
        error('NetPRS:PhenotypicNetwork:NotEnoughInputs', 'ORint and Pval are required.');
    end
    ORint = Arg1;
    Pval = Arg2;
    if nargin >= 3
        Opts = Arg3;
    else
        Opts = struct();
    end
end
if isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'PThreshold', 0.05);

s = size(ORint, 1);
if ~isequal(size(ORint), [s, s]) || ~isequal(size(Pval), [s, s])
    error('NetPRS:PhenotypicNetwork:SizeMismatch', 'ORint and Pval must be s x s matrices.');
end
upperMask = triu(true(s), 1);
Tp = nan(s, s);
h = ORint(upperMask);
t = nan(size(h));
valid = isfinite(h) & h > 0;
t(valid) = 1 - exp(-abs(log(h(valid))));
Tp(upperMask) = t;
keep = upperMask & (Pval < Opts.PThreshold) & isfinite(Tp);
Wu = zeros(s, s);
Wu(keep) = Tp(keep);
Wp = Wu + Wu';

Info.Tp = Tp;
Info.NumEdge = nnz(keep);
Info.Density = Info.NumEdge / (s * (s - 1) / 2);
Info.PThreshold = Opts.PThreshold;
end
