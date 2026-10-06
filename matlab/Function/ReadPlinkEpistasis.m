function Epi = ReadPlinkEpistasis(File, SNPIDs)
%READPLINKEPISTASIS Read a PLINK --epistasis report (.epi.cc) into matrices.
%
%   Epi = ReadPlinkEpistasis(File, SNPIDs)
%
%   The report has a header line and one line per tested pair with the
%   columns CHR1 SNP1 CHR2 SNP2 OR_INT STAT P (PLINK 1.9 --epistasis on
%   case/control data; run with '--epi1 1' to report every pair, or with
%   '--epi1 0.05' to report the pairs used for W_P). Columns are located by
%   name, so additional columns are allowed.
%
%   Inputs
%     File    path of the .epi.cc file
%     SNPIDs  {s x 1} SNP order of the model
%   Output (input of PhenotypicNetwork)
%     Epi.ORint, Epi.Stat, Epi.P  [s x s] symmetric (NaN = not reported)
%     Epi.Pairs [B x 2] index pairs read; Epi.NumSkipped lines whose SNPs
%     are not in SNPIDs

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
