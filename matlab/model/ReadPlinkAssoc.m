function Assoc = ReadPlinkAssoc(File, Test)

if nargin < 2 || isempty(Test)
    Test = 'ADD';
end
fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadPlinkAssoc:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
header = fgetl(fid);
if ~ischar(header)
    error('NetPRS:ReadPlinkAssoc:Empty', 'Empty file %s.', File);
end
cols = regexp(strtrim(header), '\s+', 'split');
C = textscan(fid, repmat('%s ', 1, numel(cols)));
col = @(name) find(strcmpi(cols, name), 1);
if isempty(col('SNP')) || isempty(col('P'))
    error('NetPRS:ReadPlinkAssoc:Column', 'Columns SNP and P are required in %s.', File);
end
rows = true(numel(C{col('SNP')}), 1);
if ~isempty(col('TEST'))
    rows = strcmpi(C{col('TEST')}, Test);
end
Assoc.SNP = C{col('SNP')}(rows);
Assoc.P = str2double(C{col('P')}(rows));
Assoc.CHR = GetText(C, col('CHR'), rows);
Assoc.A1 = GetText(C, col('A1'), rows);
Assoc.BP = GetNumber(C, col('BP'), rows);
Assoc.NMISS = GetNumber(C, col('NMISS'), rows);
Assoc.Stat = GetNumber(C, col('STAT'), rows);
if ~isempty(col('OR'))
    Assoc.OR = GetNumber(C, col('OR'), rows);
    Assoc.Beta = log(Assoc.OR);
elseif ~isempty(col('BETA'))
    Assoc.Beta = GetNumber(C, col('BETA'), rows);
    Assoc.OR = exp(Assoc.Beta);
else
    Assoc.OR = nan(nnz(rows), 1);
    Assoc.Beta = nan(nnz(rows), 1);
end
end

function v = GetNumber(C, idx, rows)
if isempty(idx)
    v = nan(nnz(rows), 1);
else
    v = str2double(C{idx}(rows));
end
end

function v = GetText(C, idx, rows)
if isempty(idx)
    v = repmat({''}, nnz(rows), 1);
else
    v = C{idx}(rows);
end
end
