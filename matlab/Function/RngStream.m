function S = RngStream(Seed, Stream)
%RNGSTREAM Portable random stream (identical in MATLAB, GNU Octave and Python).
%
%   S = RngStream(Seed)
%   S = RngStream(Seed, Stream)
%
%   Creates a bank of 64 xorshift64 lanes identified by the pair
%   (Seed, Stream). The same pair gives bit-identical numbers in MATLAB,
%   GNU Octave and the Python version (netprs.rng.RandomStream), which makes
%   the synthetic data, the cross-validation folds and the initialization of
%   beta reproducible across the two implementations and across platforms.
%
%   Specification
%     lane constants  c(1) = 0x9E3779B97F4A7C15, c(l+1) = xorshift64(c(l))
%     key             Seed + Stream * 2^32
%     state           c XOR key (a zero state is replaced by c), followed by
%                     64 warm-up steps of every lane
%   Draw numbers with RngUniform and RngPermutation.
%
%   Inputs
%     Seed    integer in [0, 2^32)
%     Stream  integer in [0, 2^16) separating uses of one seed (default 0)
%   Output
%     S       struct with field State [64 x 1] uint64

if nargin < 2 || isempty(Stream)
    Stream = 0;
end
if ~isscalar(Seed) || ~isnumeric(Seed) || Seed < 0 || Seed >= 2^32 || Seed ~= fix(Seed)
    error('NetPRS:RngStream:Seed', 'Seed must be an integer in [0, 2^32).');
end
if ~isscalar(Stream) || ~isnumeric(Stream) || Stream < 0 || Stream >= 2^16 || Stream ~= fix(Stream)
    error('NetPRS:RngStream:Stream', 'Stream must be an integer in [0, 2^16).');
end
numLane = 64;
c = zeros(numLane, 1, 'uint64');
c(1) = bitor(bitshift(uint64(2654435769), 32), uint64(2135587861));   % 0x9E3779B97F4A7C15
for lane = 2:numLane
    c(lane) = RngXorshift64(c(lane - 1));
end
key = bitor(uint64(double(Seed)), bitshift(uint64(double(Stream)), 32));
state = bitxor(c, key);
zeroState = state == 0;
state(zeroState) = c(zeroState);
for k = 1:64
    state = RngXorshift64(state);
end
S = struct('State', state);
end
