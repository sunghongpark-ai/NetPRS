function [Keep, QC] = SNPQualityControl(X, Y, Opts)

if nargin < 3 || isempty(Opts)
    Opts = struct();
end
Opts = SetDefaultField(Opts, 'MinCallRate', 0.99);
Opts = SetDefaultField(Opts, 'MinMAF', 0.05);
Opts = SetDefaultField(Opts, 'MinHWEP', 1e-6);
Opts = SetDefaultField(Opts, 'HWESamples', 'controls');

X = double(X);
[s, n] = size(X);
Y = double(Y(:)');
if numel(Y) ~= n
    error('NetPRS:SNPQualityControl:SizeMismatch', 'Y must have one entry per column of X.');
end
obs = isfinite(X);
nonInt = obs & (X ~= round(X) | X < 0 | X > 2);
if any(nonInt(:))
    error('NetPRS:SNPQualityControl:Dosage', 'X must contain hard genotype calls 0/1/2 (or NaN).');
end

QC.CallRate = sum(obs, 2) / n;
Xz = X;
Xz(~obs) = 0;
af = sum(Xz, 2) ./ (2 * sum(obs, 2));
QC.MAF = min(af, 1 - af);

switch lower(Opts.HWESamples)
    case 'controls'
        cols = (Y == 0);
    case 'all'
        cols = true(1, n);
    otherwise
        error('NetPRS:SNPQualityControl:HWESamples', 'HWESamples must be ''controls'' or ''all''.');
end
Xh = X(:, cols);
n0 = sum(Xh == 0, 2);
n1 = sum(Xh == 1, 2);
n2 = sum(Xh == 2, 2);
QC.HWEP = HWExactTest(n0, n1, n2);

QC.PassCall = (1 - QC.CallRate) <= (1 - Opts.MinCallRate) + 1e-12;
QC.PassMAF = QC.MAF >= Opts.MinMAF;
QC.PassHWE = ~(QC.HWEP < Opts.MinHWEP);
QC.PassHWE(isnan(QC.HWEP)) = false;
Keep = QC.PassCall & QC.PassMAF & QC.PassHWE;
QC.Options = Opts;
end
