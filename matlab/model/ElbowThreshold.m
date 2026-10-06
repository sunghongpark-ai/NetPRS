function [Tau, Info] = ElbowThreshold(Values, Opts)

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
