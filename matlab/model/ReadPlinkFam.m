function Fam = ReadPlinkFam(File)

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
