function [Wp, Info] = PhenotypicNetwork(Arg1, Arg2, Arg3)
%PHENOTYPICNETWORK Phenotypic SNP interaction network W_P (Sec. II-A).
%
%   [Wp, Info] = PhenotypicNetwork(ORint, Pval)
%   [Wp, Info] = PhenotypicNetwork(ORint, Pval, Opts)
%   [Wp, Info] = PhenotypicNetwork(Epi)          Epi = output of EpistasisTest
%   [Wp, Info] = PhenotypicNetwork(Epi, Opts)    or of ReadPlinkEpistasis
%
%   The interaction odds ratio h_P^{jk} = OR_INT of the epistasis test of
%   Eq. (1) is converted into a similarity by
%       T_P(x) = -exp(-|ln x|) + 1,        W_P^{jk} = T_P(h_P^{jk}),
%   (T_P(1) = 0: no interaction; T_P -> 1 as OR_INT deviates from 1; and
%   T_P(x) = T_P(1/x), so W_P does not depend on the counted alleles).
%   Only interactions with P < Opts.PThreshold are kept (0.05 in the paper).
%
%   Inputs
%     ORint, Pval      [s x s] interaction odds ratios and P-values; only the
%                      strict upper triangle is read (NaN = untested)
%     Opts.PThreshold  significance threshold (default 0.05)
%
%   Outputs
%     Wp    [s x s] symmetric, non-negative, zero diagonal
%     Info  .Tp (T_P of every tested pair; upper triangle), .NumEdge,
%           .Density = NumEdge / (s(s-1)/2), .PThreshold

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
