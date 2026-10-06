function SetRandomSeed(Seed)

if ~isscalar(Seed) || ~isnumeric(Seed) || Seed < 0 || Seed ~= fix(Seed)
    error('NetPRS:SetRandomSeed:InvalidSeed', 'Seed must be a non-negative integer.');
end
rng(double(Seed), 'twister');
end
