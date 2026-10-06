function [Data, Truth] = GenerateSyntheticData(Opts)

if nargin < 1 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'NumSubject', 1000);
Opts = SetDefaultField(Opts, 'NumValidation', 300);
Opts = SetDefaultField(Opts, 'NumSNP', 300);
Opts = SetDefaultField(Opts, 'NumGene', 120);
Opts = SetDefaultField(Opts, 'NumModule', 6);
Opts = SetDefaultField(Opts, 'Seed', 20261006);
n = Opts.NumSubject;
s = Opts.NumSNP;
m = Opts.NumGene;
K = Opts.NumModule;
if ~(Opts.NumValidation >= 0 && Opts.NumValidation < n && K >= 2 && m >= 2 * K && s >= 30)
    error('NetPRS:GenerateSyntheticData:Size', ['Invalid synthetic data dimensions (two risk modules ', ...
        'need NumModule >= 2 and NumGene >= 2 * NumModule).']);
end
intercept = -1.0;
majorEffect = 1.0;
networkEffect = 0.5;
networkMu = 1.0;
numSeedGene = 2;
causalNetworkEffect = 0.05;

module = floor((0:m-1)' * K / m) + 1;
[jj, ii] = find(triu(true(m), 1)');
S = RngStream(Opts.Seed, 11);
[uEdge, S] = RngUniform(S, numel(ii));
uWeight = RngUniform(S, numel(ii));
same = module(ii) == module(jj);
pEdge = 0.01 * ones(size(ii));
pEdge(same) = 0.30;
edge = uEdge < pEdge;
w = 0.15 + 0.35 * uWeight;
w(same) = 0.4 + 0.6 * uWeight(same);
w = RoundDigits(w, 4);
U = sparse(ii(edge), jj(edge), w(edge), m, m);
Wggi = U + U';

u = RngUniform(RngStream(Opts.Seed, 12), 4 * s);
u = reshape(u, 4, s)';
hasRel = u(:, 1) >= 0.10;
gene1 = floor(u(:, 2) * m) + 1;
hasSecond = hasRel & (u(:, 3) < 0.20);
relSNP = zeros(0, 1);
relGene = zeros(0, 1);
for j = 1:s
    if ~hasRel(j)
        continue;
    end
    relSNP(end + 1, 1) = j;
    relGene(end + 1, 1) = gene1(j);
    if hasSecond(j)
        cand = find(module == module(gene1(j)) & (1:m)' ~= gene1(j));
        relSNP(end + 1, 1) = j;
        relGene(end + 1, 1) = cand(floor(u(j, 4) * numel(cand)) + 1);
    end
end

S = RngStream(Opts.Seed, 13);
[u, S] = RngUniform(S, s);
maf = 0.05 + 0.45 * u;
[ua, S] = RngUniform(S, n * s);
ub = RngUniform(S, n * s);
Geno = double(reshape(ua, s, n) < maf) + double(reshape(ub, s, n) < maf);

S = RngStream(Opts.Seed, 14);
snpModule = zeros(s, 1);
snpModule(hasRel) = module(gene1(hasRel));
riskModule = [1, 2];
mainSNP = cell(1, 2);
mainEffect = cell(1, 2);
pairSNP = cell(1, 2);
pairEffect = cell(1, 2);
seedGene = zeros(0, 1);
for r = 1:numel(riskModule)
    cand = find(snpModule == riskModule(r) & maf > 0.15);
    if numel(cand) < 2
        error('NetPRS:GenerateSyntheticData:Size', 'Too few common SNPs in a risk module; increase NumSNP.');
    end
    [perm, S] = RngPermutation(S, numel(cand));
    main = cand(perm(1:min(8, numel(cand))));
    [ue, S] = RngUniform(S, numel(main));
    effect = 0.30 + 0.25 * ue;
    if r == 1
        effect(1) = majorEffect;
    end
    pairs = zeros(6, 2);
    effects = zeros(6, 1);
    for t = 1:6
        [perm, S] = RngPermutation(S, numel(cand));
        [uc, S] = RngUniform(S, 2);
        pairs(t, :) = [cand(perm(1)), cand(perm(2))];
        if uc(2) < 0.65
            sgn = 1;
        else
            sgn = -1;
        end
        effects(t) = (0.35 + 0.25 * uc(1)) * sgn;
    end
    mainSNP{r} = main;
    mainEffect{r} = effect;
    pairSNP{r} = pairs;
    pairEffect{r} = effects;
    genesR = find(module == riskModule(r));
    [perm, S] = RngPermutation(S, numel(genesR));
    seedGene = [seedGene; genesR(perm(1:numSeedGene))];
end
seedInd = zeros(m, 1);
seedInd(seedGene) = 1;
lap = diag(full(sum(Wggi, 2))) - full(Wggi);
geneSignal = (eye(m) + networkMu * lap) \ seedInd;
relation = zeros(m, s);
relation(sub2ind([m, s], relGene, relSNP)) = 1;
v = relation' * geneSignal;
netEffect = RoundDigits(networkEffect * v / max(v), 8);
eta = intercept * ones(1, n);
for r = 1:numel(riskModule)
    for q = 1:numel(mainSNP{r})
        j = mainSNP{r}(q);
        eta = eta + mainEffect{r}(q) * (Geno(j, :) - 2 * maf(j));
    end
    for t = 1:size(pairSNP{r}, 1)
        j = pairSNP{r}(t, 1);
        k = pairSNP{r}(t, 2);
        eta = eta + pairEffect{r}(t) * (Geno(j, :) .* Geno(k, :) - 4 * maf(j) * maf(k));
    end
end
for j = 1:s
    eta = eta + netEffect(j) * (Geno(j, :) - 2 * maf(j));
end
prob = 1 ./ (1 + exp(-eta));
uy = RngUniform(S, n);
Y = double(uy' < prob);

S = RngStream(Opts.Seed, 15);
causal = unique([mainSNP{1}; mainSNP{2}; pairSNP{1}(:); pairSNP{2}(:); find(netEffect >= causalNetworkEffect)]);
nonCausal = setdiff((1:s)', causal);
[perm, S] = RngPermutation(S, numel(nonCausal));
perm = nonCausal(perm);
defect = struct('Missing', perm(1:3), 'Rare', perm(4:6), 'HWE', perm(7:9));
for j = defect.Missing'
    [u, S] = RngUniform(S, n);
    Geno(j, u' < 0.05) = NaN;
end
for j = defect.Rare'
    [u, S] = RngUniform(S, n);
    Geno(j, :) = double(u' < 0.005);
end
for j = defect.HWE'
    [u, S] = RngUniform(S, n);
    Geno(j, :) = 1;
    Geno(j, u' < 0.05) = 0;
end
u = RngUniform(S, n * s);
Geno(reshape(u, s, n) < 0.002) = NaN;

fmtSNP = sprintf('snp%%0%dd', max(3, numel(sprintf('%d', s))));
fmtSubj = sprintf('S%%0%dd', max(4, numel(sprintf('%d', n))));
fmtGene = sprintf('GENE%%0%dd', max(3, numel(sprintf('%d', m))));
SNP = arrayfun(@(j) sprintf(fmtSNP, j), (1:s)', 'UniformOutput', false);
Gene = arrayfun(@(i) sprintf(fmtGene, i), (1:m)', 'UniformOutput', false);
Data = struct();
Data.Geno = Geno;
Data.Y = Y;
Data.IsValidation = (1:n) > n - Opts.NumValidation;
Data.SNP = SNP;
Data.A1 = {};
Data.A2 = {};
Data.IID = arrayfun(@(i) sprintf(fmtSubj, i), (1:n)', 'UniformOutput', false);
Data.Gene = Gene;
Data.Wggi = Wggi;
Data.RelSNP = SNP(relSNP);
Data.RelGene = Gene(relGene);
Data.RelWeight = ones(numel(relSNP), 1);
Data.Source = 'synthetic';
Truth = struct('GeneModule', module, 'RiskModule', riskModule, 'MainSNP', {mainSNP}, ...
    'MainEffect', {mainEffect}, 'PairSNP', {pairSNP}, 'PairEffect', {pairEffect}, ...
    'SeedGene', seedGene, 'GeneSignal', geneSignal, 'NetworkEffect', netEffect, ...
    'MAF', maf, 'Defect', defect, 'Options', Opts);
end

function y = RoundDigits(x, digits)

scale = 10 ^ digits;
y = floor(x * scale + 0.5) / scale;
end
