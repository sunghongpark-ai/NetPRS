function GWAS = MatchPlinkAssoc(Assoc, SNPID, A1, A2)
%MATCHPLINKASSOC Align a PLINK --logistic report with the genotype matrix.
%
%   GWAS = MatchPlinkAssoc(Assoc, SNPID, A1, A2)
%
%   Rows of Assoc (ReadPlinkAssoc) are matched to SNPID by identifier. The
%   log odds ratio refers to the allele Assoc.A1, whereas the genotype
%   dosage counts the .bim allele A1 (ReadPlinkBed). PLINK 1.9 re-selects A1
%   as the minor allele of the loaded samples unless --keep-allele-order is
%   given, so the sign of the effect is aligned explicitly:
%     Assoc.A1 == A1  -> Beta kept
%     Assoc.A1 == A2  -> Beta negated (OR inverted)
%     otherwise       -> Beta = NaN (allele mismatch, e.g. strand)
%   P-values do not depend on the coded allele.
%
%   Inputs
%     Assoc   struct from ReadPlinkAssoc
%     SNPID   {s x 1} SNP order of the genotype matrix
%     A1, A2  {s x 1} counted and other allele of every SNP ({} = no check)
%   Output (s x 1 vectors): GWAS.P, GWAS.Beta, GWAS.OR, GWAS.Found,
%   GWAS.Flipped

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
