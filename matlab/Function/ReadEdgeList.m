function [W, NodeIDs] = ReadEdgeList(File, NodeIDs, Opts)
%READEDGELIST Read an undirected weighted network from an edge-list file.
%
%   [W, NodeIDs] = ReadEdgeList(File)
%   [W, NodeIDs] = ReadEdgeList(File, NodeIDs, Opts)
%
%   Each line holds 'NodeA NodeB [Weight] [...]' separated by white space;
%   all lines must have the same number of fields and additional fields are
%   ignored. Typical use: the gene-gene interaction network of STRING after
%   mapping protein IDs to gene symbols, with Weight = combined_score and
%   Opts.WeightScale = 1/1000. Without a weight field every edge has weight 1.
%
%   Inputs
%     File     path of the edge list
%     NodeIDs  {m x 1} node order of the output ([] = sorted union of all
%              nodes in the file); edges with unknown nodes are skipped
%     Opts     .Header       'auto' (default) | true | false; 'auto' treats
%                            the first line as a header when its third field
%                            is not numeric
%              .WeightScale  multiplier of the weights (default 1)
%              .MinWeight    edges with scaled weight < MinWeight are
%                            dropped (default 0)
%   Outputs
%     W        [m x m] sparse symmetric adjacency with zero diagonal;
%              duplicated or reciprocal edges keep their maximum weight
%     NodeIDs  {m x 1}

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
