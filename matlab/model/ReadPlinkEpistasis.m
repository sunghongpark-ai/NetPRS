function Epi = ReadPlinkEpistasis(File, SNPIDs)

fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadPlinkEpistasis:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
header = fgetl(fid);
if ~ischar(header)
    error('NetPRS:ReadPlinkEpistasis:Empty', 'Empty file %s.', File);
end
cols = regexp(strtrim(header), '\s+', 'split');
need = {'SNP1', 'SNP2', 'OR_INT', 'STAT', 'P'};
pos = zeros(1, numel(need));
for i = 1:numel(need)
    hit = find(strcmpi(cols, need{i}), 1);
    if isempty(hit)
        error('NetPRS:ReadPlinkEpistasis:Column', ...
            'Column %s not found in %s (OR_INT requires PLINK --epistasis).', need{i}, File);
    end
    pos(i) = hit;
end
C = textscan(fid, repmat('%s ', 1, numel(cols)));
snp1 = C{pos(1)};
snp2 = C{pos(2)};
orInt = str2double(C{pos(3)});
stat = str2double(C{pos(4)});
pv = str2double(C{pos(5)});

SNPIDs = cellstr(SNPIDs);
s = numel(SNPIDs);
[ok1, j1] = ismember(snp1, SNPIDs);
[ok2, j2] = ismember(snp2, SNPIDs);
ok = ok1 & ok2 & (j1 ~= j2);
Epi.Pairs = [j1(ok), j2(ok)];
Epi.NumSkipped = nnz(~ok);
Epi.ORint = ToMatrix(Epi.Pairs, orInt(ok), s);
Epi.Stat = ToMatrix(Epi.Pairs, stat(ok), s);
Epi.P = ToMatrix(Epi.Pairs, pv(ok), s);
end

function A = ToMatrix(pairs, v, s)
A = nan(s, s);
if isempty(pairs)
    return;
end
A(sub2ind([s, s], pairs(:, 1), pairs(:, 2))) = v;
A(sub2ind([s, s], pairs(:, 2), pairs(:, 1))) = v;
end
