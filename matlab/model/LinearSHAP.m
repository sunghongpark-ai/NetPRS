function [Phi, Base] = LinearSHAP(Beta, Zexplain, Zbackground)

Beta = Beta(:);
if size(Zexplain, 1) ~= numel(Beta) || size(Zbackground, 1) ~= numel(Beta)
    error('NetPRS:LinearSHAP:SizeMismatch', 'Z matrices must have numel(Beta) rows.');
end
mu = mean(Zbackground, 2);
Phi = Beta .* (Zexplain - mu);
Base = Beta' * mu;
end
