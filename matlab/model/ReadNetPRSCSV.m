function Data = ReadNetPRSCSV(File)

fid = fopen(File, 'r', 'n', 'UTF-8');
if fid < 0
    error('NetPRS:ReadNetPRSCSV:Open', 'Cannot open %s.', File);
end
txt = fread(fid, Inf, '*char')';
fclose(fid);

if ~isempty(txt) && double(txt(1)) == 65279
    txt = txt(2:end);
elseif numel(txt) >= 3 && isequal(double(txt(1:3)), [239, 187, 191])
    txt = txt(4:end);
end
txt(txt == char(13)) = [];
if isempty(txt) || txt(end) ~= char(10)
    txt = [txt, char(10)];
end
isNL = (txt == char(10));
lineNo = cumsum([1, isNL(1:end-1)]);
numLine = lineNo(end);
content = CountPerLine(lineNo(~isspace(txt)), numLine) > 0;
contentLine = find(content);
if isempty(contentLine)
    error('NetPRS:ReadNetPRSCSV:Empty', '%s is empty.', File);
end
headerLine = contentLine(1);
dataLine = contentLine(2:end);

header = regexp(txt(lineNo == headerLine & ~isNL), ',', 'split');
header = lower(strtrim(regexprep(header, '^\s*"(.*)"\s*$', '$1')));
if ~isequal(header, {'table', 'id', 'key', 'value'})
    error('NetPRS:ReadNetPRSCSV:Header', '%s: the header must be ''table,id,key,value''.', File);
end
if isempty(dataLine)
    error('NetPRS:ReadNetPRSCSV:NoSubject', '%s: no subject rows.', File);
end
numComma = CountPerLine(lineNo(txt == ','), numLine);
bad = find(numComma(dataLine) ~= 3, 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:Fields', '%s, line %d: expected 4 fields, found %d.', ...
        File, dataLine(bad), numComma(dataLine(bad)) + 1);
end

isData = false(numLine, 1);
isData(dataLine) = true;
body = txt(reshape(isData(lineNo), 1, []));
C = textscan(body, '%s%s%s%s', 'Delimiter', ',', 'Whitespace', '', ...
    'EndOfLine', '\n', 'ReturnOnError', false);
numRow = numel(dataLine);
if any(cellfun('length', C) ~= numRow)
    error('NetPRS:ReadNetPRSCSV:Parse', '%s could not be parsed (check the field separators).', File);
end
F = [C{:}];
if any(body == ' ' | body == char(9))
    F = strtrim(F);
end
if any(body == '"')
    F = regexprep(F, '^"(.*)"$', '$1');
    q = find(~cellfun('isempty', strfind(F, '"')), 1);
    if ~isempty(q)
        [r, ~] = ind2sub(size(F), q);
        error('NetPRS:ReadNetPRSCSV:Quote', '%s, line %d: double quotes inside a field are not supported.', ...
            File, dataLine(r));
    end
end

[tableText, ~, tableIdx] = unique(F(:, 1));
[known, code] = ismember(lower(tableText), {'subject', 'genotype', 'snp_gene', 'gene_gene'});
rowTable = code(tableIdx(:));
bad = find(~known(tableIdx(:)), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:Table', '%s, line %d: unknown table %s.', File, dataLine(bad), Q(F{bad, 1}));
end
bad = find(cellfun('isempty', F(:, 2)) | cellfun('isempty', F(:, 3)), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:EmptyID', '%s, line %d: empty id or key.', File, dataLine(bad));
end

row = find(rowTable == 1);
key = lower(F(row, 3));
isDx = strcmp(key, 'diagnosis');
isVa = strcmp(key, 'validation');
bad = find(~isDx & ~isVa, 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:SubjectKey', '%s, line %d: unknown subject key %s.', ...
        File, dataLine(row(bad)), Q(F{row(bad), 3}));
end
value = Num(F(row, 4));
bad = find(value ~= 0 & value ~= 1 | isnan(value), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:SubjectValue', '%s, line %d: subject %s must be 0 or 1 (found %s).', ...
        File, dataLine(row(bad)), key{bad}, Q(F{row(bad), 4}));
end
IID = StableUnique(F(row, 2));
if isempty(IID)
    error('NetPRS:ReadNetPRSCSV:NoSubject', '%s: no subject rows.', File);
end
n = numel(IID);
Y = KeyedValue(F(row(isDx), 2), value(isDx), dataLine(row(isDx)), 'diagnosis', IID, File);
missingDx = find(isnan(Y), 1);
if ~isempty(missingDx)
    error('NetPRS:ReadNetPRSCSV:NoDiagnosis', '%s: subject ''%s'' has no diagnosis.', File, IID{missingDx});
end
isValidation = KeyedValue(F(row(isVa), 2), value(isVa), dataLine(row(isVa)), 'validation', IID, File) == 1;

row = find(rowTable == 2);
if isempty(row)
    error('NetPRS:ReadNetPRSCSV:NoGenotype', '%s: no genotype rows.', File);
end
[known, si] = ismember(F(row, 2), IID);
bad = find(~known, 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:UnknownSubject', '%s, line %d: genotype of unknown subject ''%s''.', ...
        File, dataLine(row(bad)), F{row(bad), 2});
end
[SNP, sj] = StableUnique(F(row, 3));
s = numel(SNP);
lin = sj(:) + (si(:) - 1) * s;
dup = FirstDuplicate(lin);
if ~isempty(dup)
    error('NetPRS:ReadNetPRSCSV:DuplicatedGenotype', '%s, line %d: duplicated genotype (%s, %s).', ...
        File, dataLine(row(dup)), F{row(dup), 2}, F{row(dup), 3});
end
[valueText, ~, vi] = unique(F(row, 4));
isMissing = cellfun('isempty', valueText) | strcmpi(valueText, 'NA') | strcmpi(valueText, 'NaN');
dose = Num(valueText);
dose(isMissing) = NaN;
invalid = ~isMissing & ~(dose == 0 | dose == 1 | dose == 2);
bad = find(invalid(vi(:)), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:GenotypeValue', '%s, line %d: genotype must be 0, 1, 2 or empty (found %s).', ...
        File, dataLine(row(bad)), Q(F{row(bad), 4}));
end
Geno = nan(s, n);
Geno(lin) = dose(vi(:));

row = find(rowTable == 3);
RelSNP = F(row, 2);
RelGene = F(row, 3);
RelWeight = Num(F(row, 4));
bad = find(~(isfinite(RelWeight) & RelWeight > 0), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:RelationWeight', '%s, line %d: snp_gene weight must be positive (found %s).', ...
        File, dataLine(row(bad)), Q(F{row(bad), 4}));
end

row = find(rowTable == 4);
geneA = F(row, 2);
geneB = F(row, 3);
w = Num(F(row, 4));
bad = find(~(isfinite(w) & w >= 0), 1);
if ~isempty(bad)
    error('NetPRS:ReadNetPRSCSV:GGIWeight', '%s, line %d: gene_gene weight must be >= 0 (found %s).', ...
        File, dataLine(row(bad)), Q(F{row(bad), 4}));
end
Gene = unique([RelGene(:); geneA(:); geneB(:)]);
Gene = reshape(Gene, [], 1);

Data = struct();
Data.Geno = Geno;
Data.Y = reshape(Y, 1, []);
Data.IsValidation = reshape(isValidation, 1, []);
Data.SNP = SNP;
Data.A1 = {};
Data.A2 = {};
Data.IID = IID;
Data.Gene = Gene;
Data.Wggi = EdgeMatrix(geneA, geneB, w, Gene);
Data.RelSNP = reshape(RelSNP, [], 1);
Data.RelGene = reshape(RelGene, [], 1);
Data.RelWeight = reshape(RelWeight, [], 1);
Data.Source = 'csv';
end

function v = Num(text)

v = str2double(text);
v(~cellfun('isempty', regexp(text, '[ijIJ]', 'once'))) = NaN;
v = real(v);
end

function s = Q(text)

s = ['''', text, ''''];
end

function c = CountPerLine(lineIdx, numLine)

c = accumarray(lineIdx(:), ones(numel(lineIdx), 1), [numLine, 1]);
end

function [U, Idx] = StableUnique(X)

X = X(:);
if isempty(X)
    U = cell(0, 1);
    Idx = zeros(0, 1);
    return;
end
[Us, ~, g] = unique(X);
g = g(:);
[gs, p] = sort(g);
firstPos = p([true; diff(gs) ~= 0]);
[~, order] = sort(firstPos);
U = Us(order);
U = U(:);
rank = zeros(numel(Us), 1);
rank(order) = (1:numel(Us))';
Idx = rank(g);
end

function d = FirstDuplicate(v)

[vs, p] = sort(v(:));
rep = [false; diff(vs) == 0];
if any(rep)
    d = min(p(rep));
else
    d = [];
end
end

function V = KeyedValue(ids, vals, lines, name, IID, File)

dup = FirstDuplicate(StableIndex(ids));
if ~isempty(dup)
    error('NetPRS:ReadNetPRSCSV:DuplicatedSubject', '%s, line %d: duplicated %s of subject ''%s''.', ...
        File, lines(dup), name, ids{dup});
end
[found, loc] = ismember(IID, ids);
V = nan(numel(IID), 1);
V(found) = vals(loc(found));
end

function idx = StableIndex(ids)
[~, idx] = StableUnique(ids);
end

function W = EdgeMatrix(A, B, Wt, NodeIDs)

m = numel(NodeIDs);
[okA, ia] = ismember(A(:), NodeIDs);
[okB, ib] = ismember(B(:), NodeIDs);
ok = okA & okB & (ia ~= ib) & (Wt(:) > 0);
if ~any(ok)
    W = sparse(m, m);
    return;
end
lo = min(ia(ok), ib(ok));
hi = max(ia(ok), ib(ok));
[key, ~, grp] = unique([lo, hi], 'rows');
wt = Wt(:);
wmax = accumarray(grp(:), wt(ok), [], @max);
U = sparse(key(:, 1), key(:, 2), wmax, m, m);
W = U + U';
end
