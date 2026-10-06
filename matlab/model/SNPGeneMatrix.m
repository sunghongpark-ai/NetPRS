function G = SNPGeneMatrix(SNPIDs, GeneIDs, RelSNP, RelGene, RelWeight)

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

lin = sub2ind([m, s], ig(ok), js(ok));
w = RelWeight(ok);
[ulin, ~, grp] = unique(lin);
wmax = accumarray(grp, w, [], @max);
[ii, jj] = ind2sub([m, s], ulin);
G = sparse(ii, jj, wmax, m, s);
end
