function GWAS = MatchPlinkAssoc(Assoc, SNPID, A1, A2)

SNPID = cellstr(SNPID);
s = numel(SNPID);
[found, loc] = ismember(SNPID(:), Assoc.SNP(:));
GWAS.P = nan(s, 1);
GWAS.Beta = nan(s, 1);
GWAS.Found = found;
GWAS.Flipped = false(s, 1);
GWAS.P(found) = Assoc.P(loc(found));
beta = nan(s, 1);
beta(found) = Assoc.Beta(loc(found));
if isempty(A1)
    GWAS.Beta = beta;
else
    A1 = cellstr(A1);
    A2 = cellstr(A2);
    assocA1 = repmat({''}, s, 1);
    assocA1(found) = Assoc.A1(loc(found));
    same = found & strcmpi(assocA1(:), A1(:));
    swapped = found & strcmpi(assocA1(:), A2(:)) & ~same;
    GWAS.Beta(same) = beta(same);
    GWAS.Beta(swapped) = -beta(swapped);
    GWAS.Flipped = swapped;
    bad = found & ~same & ~swapped;
    GWAS.P(bad) = NaN;
    if any(bad)
        warning('NetPRS:MatchPlinkAssoc:AlleleMismatch', ...
            '%d SNPs have an allele that matches neither A1 nor A2; they are excluded.', nnz(bad));
    end
end
GWAS.OR = exp(GWAS.Beta);
end
