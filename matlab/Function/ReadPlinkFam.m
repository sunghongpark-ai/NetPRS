function Fam = ReadPlinkFam(File)
%READPLINKFAM Read a PLINK .fam file (sample information).
%
%   Fam = ReadPlinkFam(File)
%
%   Output fields ([n x 1]): FID, IID, PAT, MAT (cellstr), SEX, PHENO, and
%   Y = binary diagnosis from the PLINK case/control coding
%   (2 = case -> 1, 1 = control -> 0, otherwise NaN).

fid = fopen(File, 'r');
if fid < 0
    error('NetPRS:ReadPlinkFam:Open', 'Cannot open %s.', File);
end
cleaner = onCleanup(@() fclose(fid));
C = textscan(fid, '%s %s %s %s %f %f', 'TreatAsEmpty', {'NA'});
Fam.FID = C{1};
Fam.IID = C{2};
Fam.PAT = C{3};
Fam.MAT = C{4};
Fam.SEX = C{5};
Fam.PHENO = C{6};
Fam.Y = nan(size(Fam.PHENO));
Fam.Y(Fam.PHENO == 2) = 1;
Fam.Y(Fam.PHENO == 1) = 0;
end
