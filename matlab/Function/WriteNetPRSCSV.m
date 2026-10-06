function WriteNetPRSCSV(File, Data)
%WRITENETPRSCSV Write the NetPRS inputs to the single-file CSV format.
%
%   WriteNetPRSCSV(File, Data)
%
%   Data is a struct as returned by LoadNetPRSData, ReadNetPRSCSV or
%   GenerateSyntheticData (fields Geno, Y, IsValidation, SNP, IID and,
%   optionally, Gene, Wggi, RelSNP, RelGene, RelWeight). The file has the
%   header 'table,id,key,value' (format: see ReadNetPRSCSV) and the rows
%     subject    diagnosis and validation flag of every subject (subject order)
%     genotype   subject-major (all SNPs of the first subject, then the next);
%                a missing call (NaN) is written as an empty value
%     snp_gene   one row per SNP-gene relation (weight 1 when RelWeight is absent)
%     gene_gene  every edge of the symmetric Wggi once (upper triangle,
%                row-major order)
%   Numbers are printed with '%.10g' and lines end with LF, so the file is
%   byte-identical to the one written by netprs.data.write_netprs_csv of the
%   Python version. Everything that ReadNetPRSCSV would reject (dosages other
%   than 0/1/2, duplicated or malformed identifiers, an asymmetric Wggi) is
%   rejected here.

required = {'Geno', 'Y', 'IsValidation', 'SNP', 'IID'};
for k = 1:numel(required)
    if ~isfield(Data, required{k})
        error('NetPRS:WriteNetPRSCSV:Field', 'Data.%s is required.', required{k});
    end
end
Geno = double(Data.Geno);
[s, n] = size(Geno);
IID = cellstr(Data.IID);
IID = IID(:);
SNP = cellstr(Data.SNP);
SNP = SNP(:);
Y = double(Data.Y(:));
isVal = double(logical(Data.IsValidation(:)));
if numel(IID) ~= n || numel(SNP) ~= s || numel(Y) ~= n || numel(isVal) ~= n
    error('NetPRS:WriteNetPRSCSV:Size', ...
        'Geno must be numel(SNP) x numel(IID), with one Y and IsValidation value per subject.');
end
if any(Y ~= 0 & Y ~= 1)
    error('NetPRS:WriteNetPRSCSV:Diagnosis', 'Data.Y must be 0 or 1.');
end
observed = Geno(~isnan(Geno));
if any(observed ~= 0 & observed ~= 1 & observed ~= 2)
    error('NetPRS:WriteNetPRSCSV:Genotype', 'Data.Geno must contain the dosages 0, 1, 2 or NaN (missing).');
end
CheckIDs(IID, 'subject');
CheckIDs(SNP, 'SNP');
CheckUnique(IID, 'subject');
CheckUnique(SNP, 'SNP');

% subject rows: diagnosis and validation of every subject
ids = reshape([IID'; IID'], [], 1);
keys = repmat({'diagnosis'; 'validation'}, n, 1);
vals = reshape([NumText(Y)'; NumText(isVal)'], [], 1);
txtSubject = CSVRows({repmat({'subject'}, 2 * n, 1), ids, keys, vals});

% genotype rows, subject-major: Geno(:) runs over the SNPs of each subject
[jj, ii] = ndgrid(1:s, 1:n);
v = Geno(:);
valueText = repmat({''}, numel(v), 1);
ok = ~isnan(v);
valueText(ok) = NumText(v(ok));
txtGenotype = CSVRows({repmat({'genotype'}, numel(v), 1), IID(ii(:)), SNP(jj(:)), valueText});

% snp_gene rows
txtRelation = '';
if isfield(Data, 'RelSNP') && ~isempty(Data.RelSNP)
    relSNP = cellstr(Data.RelSNP);
    relGene = cellstr(Data.RelGene);
    relSNP = relSNP(:);
    relGene = relGene(:);
    if isfield(Data, 'RelWeight') && ~isempty(Data.RelWeight)
        relW = double(Data.RelWeight(:));
    else
        relW = ones(numel(relSNP), 1);
    end
    if numel(relGene) ~= numel(relSNP) || numel(relW) ~= numel(relSNP)
        error('NetPRS:WriteNetPRSCSV:Relation', 'RelSNP, RelGene and RelWeight must have equal length.');
    end
    if any(~(relW > 0) | ~isfinite(relW))
        error('NetPRS:WriteNetPRSCSV:Relation', 'Relation weights must be finite and positive.');
    end
    CheckIDs([relSNP; relGene], 'relation');
    txtRelation = CSVRows({repmat({'snp_gene'}, numel(relSNP), 1), relSNP, relGene, NumText(relW)});
end

% gene_gene rows: upper triangle in row-major order
txtGGI = '';
if isfield(Data, 'Wggi') && ~isempty(Data.Wggi) && nnz(Data.Wggi) > 0
    Gene = cellstr(Data.Gene);
    Gene = Gene(:);
    W = sparse(double(Data.Wggi));
    if ~isequal(size(W), [numel(Gene), numel(Gene)])
        error('NetPRS:WriteNetPRSCSV:GGI', 'Wggi must be numel(Gene) x numel(Gene).');
    end
    w = nonzeros(W);
    if any(w < 0) || any(~isfinite(w))
        error('NetPRS:WriteNetPRSCSV:GGI', 'Wggi must be finite and non-negative.');
    end
    if nnz(W - W') > 0 && max(abs(nonzeros(W - W'))) > 1e-12 * max(abs(w))
        error('NetPRS:WriteNetPRSCSV:GGI', 'Wggi must be symmetric (each edge is written once).');
    end
    CheckIDs(Gene, 'gene');
    CheckUnique(Gene, 'gene');
    [J, I, V] = find(triu(W, 1)');                   % transposed: row-major order
    txtGGI = CSVRows({repmat({'gene_gene'}, numel(V), 1), Gene(I), Gene(J), NumText(full(V))});
end

fid = fopen(File, 'w', 'n', 'UTF-8');
if fid < 0
    error('NetPRS:WriteNetPRSCSV:Open', 'Cannot open %s for writing.', File);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, '%s', [sprintf('table,id,key,value\n'), txtSubject, txtGenotype, txtRelation, txtGGI]);
end

%% ------------------------------------------------------------------------
function T = NumText(v)
% '%.10g' text of every element of v (column cellstr); unique values only.
[u, ~, g] = unique(v(:));
ut = arrayfun(@(x) sprintf('%.10g', x), u, 'UniformOutput', false);
T = reshape(ut(g), [], 1);
end

function CheckIDs(ids, what)
bad = ~cellfun('isempty', regexp(ids, '[,"\n\r]|^\s|\s$', 'once')) | cellfun('isempty', ids);
if any(bad)
    msg = ['The ', what, ' identifier ''', ids{find(bad, 1)}, ''' is empty, contains a comma, a ', ...
        'double quote or a line break, or starts/ends with white space.'];
    error('NetPRS:WriteNetPRSCSV:ID', '%s', msg);
end
end

function CheckUnique(ids, what)
if numel(unique(ids)) ~= numel(ids)
    error('NetPRS:WriteNetPRSCSV:Duplicated', 'Duplicated %s identifiers.', what);
end
end
