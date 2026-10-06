function L = GraphLaplacian(W, Type)

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
