function [Data, Truth] = GenerateSyntheticData(Opts)
%GENERATESYNTHETICDATA Synthetic NetPRS data (generator of dataset/sample.csv).
%
%   [Data, Truth] = GenerateSyntheticData()
%   [Data, Truth] = GenerateSyntheticData(Opts)
%
%   SYNTHETIC DATA ONLY. The cohorts of the paper (ADNI, BICWALZS) and the
%   dbSNP/STRING extracts cannot be redistributed; this generator reproduces
%   their structure (genotypes of a discovery and a validation cohort, AD
%   diagnosis, SNP-gene relations, a modular gene-gene interaction network)
%   so that the whole pipeline can be executed. It says nothing about AD.
%
%   All random numbers come from the portable streams RngStream(Seed, 11..15)
%   in a fixed order. Apart from element-wise arithmetic, only one linear
%   solve and one matrix product are used, and their results are rounded to
%   8 decimals, so MATLAB, GNU Octave and the Python version
%   (netprs.synthetic) generate the same data and, with WriteNetPRSCSV, a
%   byte-identical CSV file (verified in Octave and Python).
%
%   Generative model (defaults in brackets)
%     * Genes [120] form equal-sized modules [6]. A gene pair is linked with
%       probability 0.30 within a module (weight U(0.4, 1)) and 0.01 between
%       modules (weight U(0.15, 0.5)); weights are rounded to 4 decimals.
%     * Every SNP [300] is related to one random gene with probability 0.9
%       and, with probability 0.2, to a second gene of the same module.
%     * Genotypes: independent Binomial(2, MAF), MAF ~ U(0.05, 0.5).
%     * Diagnosis: logit P(AD) = -1 + sum a_j (x_j - 2 p_j)
%       + sum c_jk (x_j x_k - 4 p_j p_k) + sum d_j (x_j - 2 p_j) (expected,
%       not sample, centring); the three terms mirror the three effects of
%       NetPRS:
%         independent effects   8 main-effect SNPs in each of two risk
%                               modules (a ~ U(0.30, 0.55); the first one is
%                               a major locus with a = 1, like APOE e4)
%         phenotypic interaction 6 epistatic SNP pairs per risk module
%                               (|c| ~ U(0.35, 0.60), 65% positive)
%         genomic interaction   2 seed genes per risk module whose signal is
%                               propagated on the gene-gene network by the
%                               GSSL closed form of Eq. (3), e = (I + L)^{-1} s
%                               (s = seed indicator, L = D - W), and passed to
%                               the SNPs through the SNP-gene relations:
%                               d = 0.5 * g' e / max(g' e), rounded to 8
%                               decimals (SNPs of interacting genes share
%                               effects)
%     * Quality-control defects among non-causal SNPs (no main or epistatic
%       effect, network effect below 0.05): 3 SNPs with 5% missing calls,
%       3 rare SNPs, 3 SNPs violating HWE; then 0.2% missing calls.
%     * The last NumValidation subjects form the validation cohort.
%
%   Opts: NumSubject 1000, NumValidation 300, NumSNP 300, NumGene 120,
%         NumModule 6, Seed 2024
%
%   Outputs
%     Data   struct of LoadNetPRSData (Geno [s x n], Y, IsValidation, SNP,
%            IID, Gene, Wggi, RelSNP, RelGene, RelWeight, A1, A2, Source)
%     Truth  generating parameters (1-based indices)

if nargin < 1 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'NumSubject', 1000);
Opts = SetDefaultField(Opts, 'NumValidation', 300);
Opts = SetDefaultField(Opts, 'NumSNP', 300);
Opts = SetDefaultField(Opts, 'NumGene', 120);
Opts = SetDefaultField(Opts, 'NumModule', 6);
Opts = SetDefaultField(Opts, 'Seed', 2024);
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
networkEffect = 0.5;               % largest SNP effect of the gene-network term
networkMu = 1.0;                   % smoothness of the propagation (mu of Eq. (3))
numSeedGene = 2;                   % seed genes per risk module
causalNetworkEffect = 0.05;        % SNPs with a larger network effect get no QC defect

%% Gene-gene interaction network (stream 11)
module = floor((0:m-1)' * K / m) + 1;
[jj, ii] = find(triu(true(m), 1)');               % upper triangle, row-major order
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

%% SNP-gene relations (stream 12)
u = RngUniform(RngStream(Opts.Seed, 12), 4 * s);
u = reshape(u, 4, s)';                            % row j: the 4 draws of SNP j
hasRel = u(:, 1) >= 0.10;
gene1 = floor(u(:, 2) * m) + 1;
hasSecond = hasRel & (u(:, 3) < 0.20);
relSNP = zeros(0, 1);
relGene = zeros(0, 1);
for j = 1:s
    if ~hasRel(j)
        continue;
    end
    relSNP(end + 1, 1) = j; %#ok<AGROW>
    relGene(end + 1, 1) = gene1(j); %#ok<AGROW>
    if hasSecond(j)
        cand = find(module == module(gene1(j)) & (1:m)' ~= gene1(j));
        relSNP(end + 1, 1) = j; %#ok<AGROW>
        relGene(end + 1, 1) = cand(floor(u(j, 4) * numel(cand)) + 1); %#ok<AGROW>
    end
end

%% Genotypes (stream 13)
S = RngStream(Opts.Seed, 13);
[u, S] = RngUniform(S, s);
maf = 0.05 + 0.45 * u;
[ua, S] = RngUniform(S, n * s);
ub = RngUniform(S, n * s);
Geno = double(reshape(ua, s, n) < maf) + double(reshape(ub, s, n) < maf);

%% Diagnosis (stream 14)
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
        effect(1) = majorEffect;                   % one APOE-like major locus
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
    seedGene = [seedGene; genesR(perm(1:numSeedGene))]; %#ok<AGROW>
end
seedInd = zeros(m, 1);
seedInd(seedGene) = 1;
lap = diag(full(sum(Wggi, 2))) - full(Wggi);
geneSignal = (eye(m) + networkMu * lap) \ seedInd;                % Eq. (3)
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

%% Quality-control defects and missing calls (stream 15)
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

%% Output
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
% Round half up to the given number of decimals (identical in Python).
scale = 10 ^ digits;
y = floor(x * scale + 0.5) / scale;
end
