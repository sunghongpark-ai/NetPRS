function P = HWExactTest(NumHom1, NumHet, NumHom2, Opts)
%HWEXACTTEST Hardy-Weinberg equilibrium exact test (Wigginton et al., 2005).
%
%   P = HWExactTest(NumHom1, NumHet, NumHom2)
%   P = HWExactTest(NumHom1, NumHet, NumHom2, Opts)
%
%   Conditional on the number of genotyped subjects N and the allele count
%   nA, the heterozygote count h has the exact distribution
%     Pr(h) = N! / (nAA! h! nBB!) * 2^h * nA! nB! / (2N)!,
%   nAA = (nA - h)/2, nBB = (nB - h)/2, h = nA (mod 2). The P-value is the
%   total probability of all configurations not more likely than the
%   observed one (relative tie tolerance 1e-7), as in the PLINK --hwe
%   filter. The distribution is enumerated in log space once per distinct
%   (N, minor-allele count), so genome-wide inputs are processed quickly.
%
%   Inputs
%     NumHom1, NumHet, NumHom2  vectors of genotype counts (same length)
%     Opts.MidP  mid-p correction (default false)
%   Output
%     P  column vector of P-values (NaN when N = 0)

if nargin < 4 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'MidP', false);
a = double(NumHom1(:));
h = double(NumHet(:));
b = double(NumHom2(:));
if numel(a) ~= numel(h) || numel(a) ~= numel(b)
    error('NetPRS:HWExactTest:SizeMismatch', 'Genotype count vectors must have equal length.');
end
allCounts = [a; h; b];
if any(allCounts < 0) || any(allCounts ~= fix(allCounts))
    error('NetPRS:HWExactTest:Counts', 'Genotype counts must be non-negative integers.');
end
N = a + h + b;
nA = 2 * a + h;
nMinor = min(nA, 2 * N - nA);
P = nan(size(a));
vIdx = find(N > 0);
if isempty(vIdx)
    return;
end
[ukey, ~, grp] = unique([N(vIdx), nMinor(vIdx)], 'rows');
[grpSorted, order] = sort(grp);
memberSorted = vIdx(order);
grpStart = [1; find(diff(grpSorted) ~= 0) + 1];
grpEnd = [grpStart(2:end) - 1; numel(grpSorted)];
for u = 1:size(ukey, 1)
    n = ukey(u, 1);
    na = ukey(u, 2);                                % minor-allele count
    nb = 2 * n - na;
    het = (mod(na, 2):2:na)';                      % feasible heterozygote counts
    hom1 = (na - het) / 2;
    hom2 = (nb - het) / 2;
    logp = gammaln(n + 1) - gammaln(hom1 + 1) - gammaln(het + 1) - gammaln(hom2 + 1) ...
        + het * log(2) + gammaln(na + 1) + gammaln(nb + 1) - gammaln(2 * n + 1);
    prob = exp(logp - max(logp));
    prob = prob / sum(prob);
    members = memberSorted(grpStart(u):grpEnd(u));
    [uh, ~, hg] = unique(h(members));
    pu = zeros(numel(uh), 1);
    for t = 1:numel(uh)
        obs = find(het == uh(t), 1);
        if isempty(obs)
            error('NetPRS:HWExactTest:Infeasible', 'Infeasible genotype counts.');
        end
        pObs = prob(obs);
        if Opts.MidP
            pu(t) = sum(prob(prob < pObs * (1 - 1e-7))) + 0.5 * sum(prob(abs(prob - pObs) <= pObs * 1e-7));
        else
            pu(t) = sum(prob(prob <= pObs * (1 + 1e-7)));
        end
    end
    P(members) = min(1, pu(hg));
end
end
