function Raw = ReadPlinkRaw(File)

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
