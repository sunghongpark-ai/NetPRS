function [X, Bim, Fam] = ReadPlinkBed(Prefix, SNPIndex)
%READPLINKBED Read genotypes from a PLINK 1 binary fileset (.bed/.bim/.fam).
%
%   [X, Bim, Fam] = ReadPlinkBed(Prefix)
%   [X, Bim, Fam] = ReadPlinkBed(Prefix, SNPIndex)
%
%   X(j, i) is the dosage of the first .bim allele (A1; the minor allele in
%   PLINK 1.9 output by default) of variant SNPIndex(j) in sample i:
%       code 00 -> 2 (homozygous A1), 10 -> 1 (heterozygous),
%       code 11 -> 0 (homozygous A2), 01 -> NaN (missing),
%   following the PLINK 1.9 .bed specification (magic bytes 0x6c 0x1b,
%   SNP-major mode 0x01; the low-order bit pair of each byte holds the first
%   sample of the group of four).
%
%   Inputs
%     Prefix    path of the fileset without extension
%     SNPIndex  variant indices or logical mask (default: all variants)
%   Outputs
%     X    [k x n] dosage (double, NaN = missing)
%     Bim  variant information of the selected variants (ReadPlinkBim)
%     Fam  sample information (ReadPlinkFam)

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

% Decoding table (lut): byte value -> dosages of its four samples.
code2dose = [2, NaN, 1, 0];
byteVal = (0:255)';
lut = zeros(256, 4);
for slot = 1:4
    lut(:, slot) = code2dose(bitand(floor(byteVal / 4 ^ (slot - 1)), 3) + 1)';
end
D = reshape(lut(double(raw(:)) + 1, :), nb, k, 4);      % byte x SNP x slot
D = reshape(permute(D, [3, 1, 2]), 4 * nb, k);           % sample x SNP
X = D(1:n, :)';

fields = fieldnames(Bim);
for f = 1:numel(fields)
    Bim.(fields{f}) = Bim.(fields{f})(SNPIndex);
end
end
