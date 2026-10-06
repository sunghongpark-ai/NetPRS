function [W, NodeIDs] = ReadEdgeList(File, NodeIDs, Opts)

if nargin < 2
    NodeIDs = [];
end
if nargin < 3 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'Header', 'auto');
Opts = SetDefaultField(Opts, 'WeightScale', 1);
Opts = SetDefaultField(Opts, 'MinWeight', 0);

fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadEdgeList:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
first = fgetl(fid);
if ~ischar(first)
    error('NetPRS:ReadEdgeList:Empty', 'Empty file %s.', File);
end
tok = regexp(strtrim(first), '\s+', 'split');
numField = numel(tok);
if numField < 2
    error('NetPRS:ReadEdgeList:Parse', 'Each line of %s needs at least two fields.', File);
end
if ischar(Opts.Header)
    hasHeader = numField >= 3 && isnan(str2double(tok{3}));
else
    hasHeader = logical(Opts.Header);
end
frewind(fid);
if hasHeader
    fgetl(fid);
end
if numField >= 3
    fmt = ['%s %s %f', repmat(' %*s', 1, numField - 3)];
    C = textscan(fid, fmt);
    a = C{1};
    b = C{2};
    w = C{3};
else
    C = textscan(fid, '%s %s');
    a = C{1};
    b = C{2};
    w = ones(numel(a), 1);
end
if numel(b) ~= numel(a) || numel(w) ~= numel(a)
    error('NetPRS:ReadEdgeList:Parse', 'Inconsistent number of fields in %s.', File);
end
w = double(w) * Opts.WeightScale;
if any(~isfinite(w)) || any(w < 0)
    error('NetPRS:ReadEdgeList:Weight', 'Edge weights must be finite and non-negative.');
end
if isempty(NodeIDs)
    NodeIDs = unique([a; b]);
end
NodeIDs = cellstr(NodeIDs);
NodeIDs = NodeIDs(:);
m = numel(NodeIDs);
[okA, ia] = ismember(a, NodeIDs);
[okB, ib] = ismember(b, NodeIDs);
ok = okA & okB & (ia ~= ib) & (w >= Opts.MinWeight) & (w > 0);
lo = min(ia(ok), ib(ok));
hi = max(ia(ok), ib(ok));
if isempty(lo)
    W = sparse(m, m);
    return;
end
[key, ~, grp] = unique([lo, hi], 'rows');
wmax = accumarray(grp, w(ok), [], @max);
U = sparse(key(:, 1), key(:, 2), wmax, m, m);
W = U + U';
end
