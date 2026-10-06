function Bim = ReadPlinkBim(File)
%READPLINKBIM Read a PLINK .bim file (variant information).
%
%   Bim = ReadPlinkBim(File)
%
%   Output fields ([V x 1]): CHR (cellstr), SNP (cellstr), CM, BP, A1, A2
%   (cellstr). A1 is the allele counted by ReadPlinkBed (first allele).

fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadPlinkBim:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
C = textscan(fid, '%s %s %f %f %s %s');
Bim.CHR = C{1};
Bim.SNP = C{2};
Bim.CM = C{3};
Bim.BP = C{4};
Bim.A1 = C{5};
Bim.A2 = C{6};
if numel(unique(Bim.SNP)) ~= numel(Bim.SNP)
    warning('NetPRS:ReadPlinkBim:DuplicatedID', 'Duplicated variant IDs in %s.', File);
end
end
