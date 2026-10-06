function L = GraphLaplacian(W, Type)
%GRAPHLAPLACIAN Graph Laplacian of a symmetric non-negative network.
%
%   L = GraphLaplacian(W)               unnormalized Laplacian L = D - W
%   L = GraphLaplacian(W, 'normalized') L = I - D^(-1/2) * W * D^(-1/2)
%
%   The NetPRS paper defines L = D - W with D = diag(D^a), D^a = sum_b W^ab
%   (Sec. II-B, below Eq. (2)); this is the default. The symmetric normalized
%   form is the one used in PPIxGPN/BIGPN and is provided as an option. For an
%   isolated node (degree 0) the normalized form uses D^(-1/2) = 0, exactly as
%   in PPIxGPN/BIGPN, which gives L(i,i) = 1.
%
%   Inputs
%     W     [m x m] symmetric, non-negative, finite adjacency (full or sparse)
%     Type  'unnormalized' (default) | 'normalized'
%
%   Output
%     L     [m x m] Laplacian, of the same storage class (full/sparse) as W
%
%   Self-loops cancel in D - W but not in the normalized form; the network
%   builders of this codeset return networks with a zero diagonal.

if nargin < 2 || isempty(Type)
    Type = 'unnormalized';
end
CheckNetwork(W);
m = size(W, 1);
deg = full(sum(W, 2));
switch lower(Type)
    case 'unnormalized'
        if issparse(W)
            L = spdiags(deg, 0, m, m) - W;
        else
            L = diag(deg) - W;
        end
    case 'normalized'
        invSqrtDeg = zeros(m, 1);
        pos = deg > 0;
        invSqrtDeg(pos) = 1 ./ sqrt(deg(pos));
        if issparse(W)
            Dh = spdiags(invSqrtDeg, 0, m, m);
            L = speye(m) - Dh * W * Dh;
        else
            L = eye(m) - (invSqrtDeg * invSqrtDeg') .* W;
        end
    otherwise
        error('NetPRS:GraphLaplacian:UnknownType', ...
            'Unknown Laplacian type ''%s'' (use ''unnormalized'' or ''normalized'').', Type);
end
% Remove rounding-level asymmetry so that symmetric eigensolvers are used.
L = (L + L') / 2;
end

function CheckNetwork(W)
if ~isnumeric(W) || ndims(W) ~= 2 || size(W, 1) ~= size(W, 2)
    error('NetPRS:GraphLaplacian:NotSquare', 'W must be a square numeric matrix.');
end
v = nonzeros(W);
if any(~isfinite(v))
    error('NetPRS:GraphLaplacian:NonFinite', 'W must contain finite values only.');
end
if any(v < 0)
    error('NetPRS:GraphLaplacian:Negative', 'W must be non-negative.');
end
asym = W - W';
if nnz(asym) > 0
    scale = max(abs(v));
    if max(abs(nonzeros(asym))) > 1e-12 * max(scale, 1)
        error('NetPRS:GraphLaplacian:NotSymmetric', ...
            'W must be symmetric (undirected network); symmetrize it explicitly before use.');
    end
end
end
