function X = RngXorshift64(X)

X = bitxor(X, bitshift(X, 13));
X = bitxor(X, bitshift(X, -7));
X = bitxor(X, bitshift(X, 17));
end
