function G = SNPGeneMatrix(SNPIDs, GeneIDs, RelSNP, RelGene, RelWeight)
%SNPGENEMATRIX SNP-gene relation matrix g in R^{m x s} (Sec. II-B).
%
%   G = SNPGeneMatrix(SNPIDs, GeneIDs, RelSNP, RelGene)
%   G = SNPGeneMatrix(SNPIDs, GeneIDs, RelSNP, RelGene, RelWeight)
%
%   Builds the sparse matrix with G(i, j) = 1 when SNP j is related to gene
%   i (e.g. the SNP-gene relations of dbSNP used in the paper), restricted
%   to the given SNP and gene lists. Relations whose SNP or gene is not in
%   the lists are ignored; duplicated relations are counted once (binary)
%   or, with weights, keep their maximum weight.
%
%   Inputs
%     SNPIDs     {s x 1} cellstr, SNP order of the model (rows of X)
%     GeneIDs    {m x 1} cellstr, gene order of the GGI network
%     RelSNP     {R x 1} cellstr, SNP of each relation
%     RelGene    {R x 1} cellstr, gene of each relation
%     RelWeight  [R x 1] optional non-negative weights (default 1)
%
%   Output
%     G  [m x s] sparse, non-negative

if nargin < 5 || isempty(RelWeight)
    RelWeight = ones(numel(RelSNP), 1);
end
SNPIDs = cellstr(SNPIDs);
GeneIDs = cellstr(GeneIDs);
RelSNP = cellstr(RelSNP);
RelGene = cellstr(RelGene);
RelWeight = double(RelWeight(:));
if numel(RelSNP) ~= numel(RelGene) || numel(RelSNP) ~= numel(RelWeight)
    error('NetPRS:SNPGeneMatrix:SizeMismatch', 'RelSNP, RelGene and RelWeight must have equal length.');
end
if numel(unique(SNPIDs)) ~= numel(SNPIDs) || numel(unique(GeneIDs)) ~= numel(GeneIDs)
    error('NetPRS:SNPGeneMatrix:Duplicated', 'SNPIDs and GeneIDs must be unique.');
end
if any(RelWeight < 0) || any(~isfinite(RelWeight))
    error('NetPRS:SNPGeneMatrix:Weight', 'RelWeight must be finite and non-negative.');
end
[okS, js] = ismember(RelSNP(:), SNPIDs(:));
[okG, ig] = ismember(RelGene(:), GeneIDs(:));
ok = okS & okG;
s = numel(SNPIDs);
m = numel(GeneIDs);
if ~any(ok)
    G = sparse(m, s);
    return;
end
% maximum weight of duplicated (gene, SNP) relations
lin = sub2ind([m, s], ig(ok), js(ok));
w = RelWeight(ok);
[ulin, ~, grp] = unique(lin);
wmax = accumarray(grp, w, [], @max);
[ii, jj] = ind2sub([m, s], ulin);
G = sparse(ii, jj, wmax, m, s);
end
