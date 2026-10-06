function X = RngXorshift64(X)
%RNGXORSHIFT64 One xorshift64 step (Marsaglia, 2003; shifts 13, 7, 17).
%
%   X = RngXorshift64(X) advances every element of the uint64 array X.
%   Only 64-bit shifts (overflowing bits are dropped) and exclusive-or are
%   used, so MATLAB, GNU Octave and NumPy produce identical bits.

X = bitxor(X, bitshift(X, 13));
X = bitxor(X, bitshift(X, -7));
X = bitxor(X, bitshift(X, 17));
end
