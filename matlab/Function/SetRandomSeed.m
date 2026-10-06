function SetRandomSeed(Seed)
%SETRANDOMSEED Reset the global random number generator (Mersenne Twister).
%
%   SetRandomSeed(Seed) is equivalent to the statement used in NeuroFANN and
%   BIGPN,
%       RandStream.setGlobalStream(RandStream('mt19937ar','Seed',Seed))
%   and is written with rng() so that the same code also runs in GNU Octave.
%
%   Note: the Mersenne Twister seeding of MATLAB and GNU Octave differs, so a
%   given seed reproduces the same numbers only within the same platform.

if ~isscalar(Seed) || ~isnumeric(Seed) || Seed < 0 || Seed ~= fix(Seed)
    error('NetPRS:SetRandomSeed:InvalidSeed', 'Seed must be a non-negative integer.');
end
rng(double(Seed), 'twister');
end
