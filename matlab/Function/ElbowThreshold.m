function [Tau, Info] = ElbowThreshold(Values, Opts)
%ELBOWTHRESHOLD Elbow (knee) point of a sorted value curve.
%
%   [Tau, Info] = ElbowThreshold(Values)
%   [Tau, Info] = ElbowThreshold(Values, Opts)
%
%   The values are sorted in decreasing order and plotted against their
%   normalized rank; the elbow is the point with the maximum distance to
%   the straight line joining the first and the last point (the geometric
%   "maximum distance to the chord" rule). The value at the elbow is
%   returned as threshold, so that values >= Tau are kept. This is used for
%   the genomic interaction network, whose interactions were "selected by
%   the elbow point" in the paper (Sec. IV-B).
%
%   Inputs
%     Values      vector of positive values (non-finite values are ignored)
%     Opts.Scale  'linear' (default) | 'log' (curve of log10 values, useful
%                 when the values span several orders of magnitude)
%
%   Outputs
%     Tau   threshold (value at the elbow)
%     Info  .Index (rank of the elbow), .Sorted (sorted values),
%           .Distance (normalized distance of every point to the chord)

if nargin < 2 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'Scale', 'linear');

v = Values(:);
v = v(isfinite(v));
if strcmpi(Opts.Scale, 'log')
    v = v(v > 0);
end
if isempty(v)
    error('NetPRS:ElbowThreshold:Empty', 'No valid values to threshold.');
end
v = sort(v, 'descend');
N = numel(v);
switch lower(Opts.Scale)
    case 'linear'
        yv = v;
    case 'log'
        yv = log10(v);
    otherwise
        error('NetPRS:ElbowThreshold:Scale', 'Opts.Scale must be ''linear'' or ''log''.');
end
if N < 3 || yv(1) == yv(end)
    Tau = v(end);
    Info = struct('Index', N, 'Sorted', v, 'Distance', zeros(N, 1));
    return;
end
x = (0:N-1)' / (N - 1);
y = (yv - yv(end)) / (yv(1) - yv(end));
distance = abs(1 - x - y) / sqrt(2);
[~, idx] = max(distance);
Tau = v(idx);
Info = struct('Index', idx, 'Sorted', v, 'Distance', distance);
end
