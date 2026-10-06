function Raw = ReadPlinkRaw(File)
%READPLINKRAW Read a PLINK additive genotype file (--recode A, '.raw').
%
%   Raw = ReadPlinkRaw(File)
%
%   Format: a header line 'FID IID PAT MAT SEX PHENOTYPE <ID>_<allele> ...'
%   and one line per sample with allele dosages 0/1/2/NA.
%
%   Output
%     Raw.X              [s x n] dosage of the counted allele (NaN = NA);
%                        SNPs in rows to match the NetPRS convention
%     Raw.SNP            {s x 1} variant IDs (suffix '_<allele>' removed)
%     Raw.CountedAllele  {s x 1}
%     Raw.FID, Raw.IID   {n x 1}
%     Raw.SEX, Raw.PHENO [n x 1];  Raw.Y binary diagnosis (2 -> 1, 1 -> 0)
%
%   For genome-wide data prefer ReadPlinkBed, which reads selected SNPs only.

fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadPlinkRaw:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
header = fgetl(fid);
if ~ischar(header)
    error('NetPRS:ReadPlinkRaw:Empty', 'Empty file %s.', File);
end
tok = regexp(strtrim(header), '\s+', 'split');
if numel(tok) < 7 || ~strcmpi(tok{1}, 'FID') || ~strcmpi(tok{2}, 'IID')
    error('NetPRS:ReadPlinkRaw:Header', 'Unexpected header in %s.', File);
end
snpCols = tok(7:end);
s = numel(snpCols);
fmt = ['%s %s %s %s %f %f', repmat(' %f', 1, s)];
C = textscan(fid, fmt, 'TreatAsEmpty', {'NA'}, 'CollectOutput', true);
Raw.FID = C{1}(:, 1);
Raw.IID = C{1}(:, 2);
numeric = C{2};
Raw.SEX = numeric(:, 1);
Raw.PHENO = numeric(:, 2);
Raw.X = numeric(:, 3:end)';
if size(Raw.X, 1) ~= s
    error('NetPRS:ReadPlinkRaw:Parse', 'Could not parse the genotype columns of %s.', File);
end
Raw.SNP = cell(s, 1);
Raw.CountedAllele = cell(s, 1);
for j = 1:s
    name = snpCols{j};
    cut = find(name == '_', 1, 'last');
    if isempty(cut)
        Raw.SNP{j} = name;
        Raw.CountedAllele{j} = '';
    else
        Raw.SNP{j} = name(1:cut-1);
        Raw.CountedAllele{j} = name(cut+1:end);
    end
end
Raw.Y = nan(size(Raw.PHENO));
Raw.Y(Raw.PHENO == 2) = 1;
Raw.Y(Raw.PHENO == 1) = 0;
end
