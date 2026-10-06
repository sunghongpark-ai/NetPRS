function S = RngStream(Seed, Stream)

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
c(1) = bitor(bitshift(uint64(2654435769), 32), uint64(2135587861));
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
