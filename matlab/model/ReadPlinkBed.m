function [X, Bim, Fam] = ReadPlinkBed(Prefix, SNPIndex)

Bim = ReadPlinkBim([Prefix, '.bim']);
Fam = ReadPlinkFam([Prefix, '.fam']);
numVar = numel(Bim.SNP);
n = numel(Fam.IID);
if nargin < 2 || isempty(SNPIndex)
    SNPIndex = 1:numVar;
end
if islogical(SNPIndex)
    if numel(SNPIndex) ~= numVar
        error('NetPRS:ReadPlinkBed:Index', 'Logical SNPIndex must have one entry per variant.');
    end
    SNPIndex = find(SNPIndex);
end
SNPIndex = SNPIndex(:)';
if any(SNPIndex < 1 | SNPIndex > numVar | SNPIndex ~= fix(SNPIndex))
    error('NetPRS:ReadPlinkBed:Index', 'SNPIndex out of range.');
end

fid = fopen([Prefix, '.bed'], 'r');
if fid < 0
    error('NetPRS:ReadPlinkBed:Open', 'Cannot open %s.bed.', Prefix);
end
cleaner = onCleanup(@() fclose(fid));
magic = fread(fid, 3, 'uint8=>double');
if numel(magic) < 3 || magic(1) ~= 108 || magic(2) ~= 27
    error('NetPRS:ReadPlinkBed:Format', 'Not a PLINK 1 .bed file (wrong magic number).');
end
if magic(3) ~= 1
    error('NetPRS:ReadPlinkBed:Format', 'Only SNP-major .bed files are supported.');
end
nb = ceil(n / 4);
k = numel(SNPIndex);
if k > numVar / 8
    raw = fread(fid, [nb, numVar], 'uint8=>uint8');
    if size(raw, 2) ~= numVar
        error('NetPRS:ReadPlinkBed:Truncated', 'The .bed file is truncated.');
    end
    raw = raw(:, SNPIndex);
else
    raw = zeros(nb, k, 'uint8');
    for j = 1:k
        fseek(fid, 3 + (SNPIndex(j) - 1) * nb, 'bof');
        block = fread(fid, nb, 'uint8=>uint8');
        if numel(block) ~= nb
            error('NetPRS:ReadPlinkBed:Truncated', 'The .bed file is truncated.');
        end
        raw(:, j) = block;
    end
end

code2dose = [2, NaN, 1, 0];
byteVal = (0:255)';
lut = zeros(256, 4);
for slot = 1:4
    lut(:, slot) = code2dose(bitand(floor(byteVal / 4 ^ (slot - 1)), 3) + 1)';
end
D = reshape(lut(double(raw(:)) + 1, :), nb, k, 4);
D = reshape(permute(D, [3, 1, 2]), 4 * nb, k);
X = D(1:n, :)';

fields = fieldnames(Bim);
for f = 1:numel(fields)
    Bim.(fields{f}) = Bim.(fields{f})(SNPIndex);
end
end
