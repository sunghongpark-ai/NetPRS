function Score = WeightedPRS(X, Weight)

Weight = double(Weight(:));
if size(X, 1) ~= numel(Weight)
    error('NetPRS:WeightedPRS:SizeMismatch', 'X must have numel(Weight) rows.');
end
if any(~isfinite(X(:)))
    error('NetPRS:WeightedPRS:MissingGenotype', 'X contains NaN/Inf; impute missing genotypes first.');
end
Weight(~isfinite(Weight)) = 0;
Score = Weight' * double(X);
end
