# NetPRS — MATLAB implementation (version 2)

MATLAB code of **NetPRS: SNP interaction aware network-based polygenic risk score for Alzheimer's disease** (S. Park, D. Lee, J. Kim, *et al.*, *2024 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)*, pp. 1–8, 2024, [doi:10.1109/BHI62660.2024.10913658](https://doi.org/10.1109/BHI62660.2024.10913658)).

By default the complete analysis runs on [`../dataset/sample.csv`](../dataset/README.md), a synthetic dataset in a single-file CSV format. The Python version in [`../python`](../python/README.md) implements the same code path and gives the same results.

## Quick start

```matlab
cd matlab
NetPRS                       % complete analysis of ../dataset/sample.csv -> result/
cd Test
RunAllTests                  % 39 verification tests
```

`NetPRS.m` only sets the configuration and calls three functions, so the analysis can also be scripted:

```matlab
addpath('Function');
[Config, Parameter] = NetPRSDefaults();          % defaults (dataset/sample.csv)
Config.CSVFile = 'my_cohort.csv';                % your data in the same format
Parameter.NumIter = 2;                           % quicker run
[Result, Info] = RunNetPRS(Config, Parameter);   % Stage 1 + Stage 2
ExportResults(Result, Info);                     % CSV tables, run_info.json, MAT file
```

**Requirements.** MATLAB R2016b or later without toolboxes (Parallel Computing Toolbox only for `Config.NumWorkers > 0`), or GNU Octave (tested with 8.4.0). All code and tests were executed in GNU Octave 8.4.0; MATLAB itself was not available, so please run `RunAllTests` once in MATLAB.

## What the analysis does

| Step | Function | Paper |
| :--- | :------- | :---- |
| 1. Data | `LoadNetPRSData` → `ReadNetPRSCSV` (`'csv'`), `ReadPlinkBed` & co. (`'plink'`) or `GenerateSyntheticData` (`'synthetic'`) | Table I |
| 2. SNP screening (discovery cohort) | `SNPQualityControl` (call rate > 0.99, HWE *P* > 1E−6, MAF > 5%), `ImputeGenotype`, `GWASLogistic`, `SelectSNPLevel` | Sec. IV-B |
| 3. Phenotypic network W_P | `EpistasisTest` (Eq. (1)) → `PhenotypicNetwork` (T_P, *P* < 0.05) | Sec. II-A |
| 3. Genomic network W_G | `SNPGeneMatrix`, `GenomicNetwork` (Eq. (2)–(3), T_G, elbow threshold) | Sec. II-B |
| 4. NetPRS | `ModelInitialize`, `RunCrossValidation` → `TrainNetPRS` (Eq. (4)–(6), ADAM), `SummarizeCV` | Sec. III, IV-C |
| 5. Ablation | the seven effect sets I, P, G, I+P, I+G, P+G, I+P+G | Fig. 3(b) |
| 6. Baseline | `WeightedPRS` with the discovery-cohort GWAS log odds ratios | Fig. 3(a) |
| 7. Interpretation | final model; `EffectExtraction`, `EffectSignificance` (P_X vs P_Z), `LinearSHAP` | Fig. 4 |
| 8. Export | `ExportResults` | — |

The networks are built once on the union of the analysed levels and restricted to each level (both interactions are pairwise, so this equals building them per level, and W_G has a single elbow threshold as in the paper). Every model is evaluated by 10 × stratified 5-fold cross-validation in the discovery cohort — four folds train, one fold selects the epoch with the minimum validation loss — and the validation cohort is the test set; the reported AUC is the mean test AUC of the 50 fits.

## Changes from the first version (`codeset`)

| Area | Version 2 |
| :--- | :-------- |
| Input | New single-file CSV format for genotype, diagnosis, cohort, SNP-gene relations and gene-gene interactions: `ReadNetPRSCSV` (validated, errors name the line) and `WriteNetPRSCSV` (rejects whatever the reader would reject); `'csv'` is the default data source. For CSV and PLINK input the gene set is the union of interaction and relation genes, so both inputs give the same W_G (in version 1 the PLINK path dropped relations to genes without interactions). |
| Structure | `NetPRS.m` is a short script; the analysis is the function `RunNetPRS` (returns `Result`/`Info`, no global state), defaults live in `NetPRSDefaults`, outputs in `ExportResults`. |
| Reproducibility | Folds and the initial β come from a portable random generator (`RngStream`, `RngUniform`, `RngPermutation`, `RngXorshift64`) instead of the global Mersenne Twister, whose streams differ between MATLAB and Octave. MATLAB, Octave and Python now train identical models. |
| Sample data | `GenerateSyntheticData` rewritten with the portable generator; it plants independent, epistatic and gene-network effects (one per NetPRS effect) and reproduces `dataset/sample.csv` byte for byte. |
| Robustness | Relative tie tolerance 1E−9 for the W_G threshold (`Opts.TieTol`) and for counting P_Z < P_X; validation of the configuration and of the seed range; data without SNP-gene relations run with the effects I and P; clear errors for empty cohorts; CSV text is assembled by concatenation (`CSVRows`) because MATLAB's `sprintf` skips empty arguments such as missing calls, unlike Octave. |
| Results | `SummarizeCV` keeps every fit (iteration, folds, epoch, AUCs, θ, μ); `ExportResults` writes seven files (below). |
| Tests | `RunAllTests` grew from 31 to 39 tests: random streams, CSV format, reproduction of `sample.csv`, golden values shared with Python, CSV-vs-memory equivalence, export, tie tolerances, the penalty property below. |

New functions: `ReadNetPRSCSV`, `WriteNetPRSCSV`, `CSVRows`, `RngStream`, `RngUniform`, `RngPermutation`, `RngXorshift64`, `NetPRSDefaults`, `RunNetPRS`, `ExportResults`. Modified: `LoadNetPRSData`, `GenerateSyntheticData`, `ModelInitialize`, `ParamInitialize`, `GenomicNetwork`, `EffectSignificance`, `SummarizeCV`. The numerical core (networks, propagation, gradients, ADAM, statistics, PLINK readers) is unchanged apart from the W_G tie tolerance and is still verified against independent references.

## Folder structure

```text
matlab/
├── NetPRS.m                main script (configuration -> RunNetPRS -> ExportResults)
├── README.md
├── Function/               58 functions
└── Test/RunAllTests.m      verification suite (39 tests)
```

## Configuration

`NetPRSDefaults` returns both structures; absent fields passed to `RunNetPRS` take these defaults.

| `Config` field | Default | Meaning |
| :------------- | :------ | :------ |
| `DataSource` | `'csv'` | `'csv'`, `'plink'` or `'synthetic'` |
| `CSVFile` | `<repository>/dataset/sample.csv` | input in the CSV format of [dataset/README.md](../dataset/README.md) |
| `PlinkPrefix`, `ValidationIIDFile`, `SNPGeneFile`, `GGIFile`, `GGIWeightScale` | `''`, …, `1/1000` | PLINK input (see "Using your own data") |
| `GWASFile`, `EpistasisFile` | `''` | optional PLINK `--logistic` / `--epistasis` reports of the discovery cohort |
| `RunQC`, `QCOptions` | `true`, `struct()` | quality control of the discovery cohort |
| `LevelThresholds`, `Levels` | `[1 1.5 2]`, `1:3` | −log10 *P* cut-offs (paper: ADNI `[3 4 5]`, BICWALZS `[4 5 6]`; the sample has only 300 SNPs) |
| `MinLevelSNP` | 3 | smaller levels are skipped |
| `EpistasisPThreshold` | 0.05 | W_P edges (paper) |
| `GenomicMu`, `GenomicSigma`, `GenomicThreshold` | 1, 1, `'elbow'` | μ of Eq. (3) (paper), σ of T_G, threshold of W_G |
| `Effects` | `'IPG'` | effects of the NetPRS model |
| `RunAblation`, `RunBaseline`, `RunInterpretation` | `true` | experiments of Fig. 3–4 |
| `NumKeySNP`, `NumWorkers` | 10, 0 | SNPs listed by mean \|SHAP\|; `parfor` workers |
| `OutputDir`, `Verbose` | `matlab/result`, `true` | output folder; progress messages |

| `Parameter` field | Default | Basis |
| :---------------- | :------ | :---- |
| `NumIter`, `NumFold` | 10, 5 | repeated stratified K-fold CV (NeuroFANN/BIGPN) |
| `MaxEpoch` | 500 | ADAM steps (NeuroFANN/BIGPN/PPIxGPN) |
| `LearnRate` | 0.01 | paper |
| `RegCoeff` | 0.005 | δ of Eq. (6) (NeuroFANN) |
| `MuInit`, `AlphaInit` | 1, 0 | paper |
| `MuMin` | 0 | projection μ ≥ 0 after each step (safeguard) |
| `Laplacian`, `Solver` | `'unnormalized'`, `'eig'` | L = D − W (paper); exact eigendecomposition solver (`'chol'` = reference) |
| `XScaling` | `'center'` | per-SNP centring with training-fold means |
| `Seed` | 0 | folds from `RngStream(Seed + i, 1)`, β from `RngStream(Seed + i, 2)` for iteration i |

## Outputs (`Config.OutputDir`)

| File | Content |
| :--- | :------ |
| `auc_summary.csv` | per level and model: mean and SD of the test AUC over the CV fits, ensemble AUC, number of fits, median selected epoch, mean θ_I/θ_P/θ_G and μ_P/μ_G; wPRS row |
| `cv_models.csv` | every CV fit: iteration, validation fold, selected epoch, train/validation/test AUC, θ, μ |
| `networks.csv` | Table III statistics of W_P and W_G per level (edges, mean weight, density, common edges, correlation, KS *P*) |
| `snps.csv` | every SNP: call rate, MAF, HWE *P*, QC pass, GWAS β/SE/*P*, highest level |
| `interpretation.csv` | final model per level and SNP: β, effective weight Ω, P_X, P_Z, mean \|SHAP\|, mean SHAP in cases/controls, rank |
| `final_model.csv` | final model per level: epochs, θ, μ, validation AUC, P_Z < P_X summary, SHAP base value |
| `run_info.json` | configuration, parameters, data summary, software version, run time |
| `NetPRS_Result.mat` | `Result` and `Info` structs |

Numbers are written with `%.10g` and NaN as an empty field, exactly as the Python version, so the folders of both versions can be compared with `python ../python/tools/compare_results.py result ../python/result`.

## Results on `sample.csv`

Default settings; GNU Octave 8.4.0 on 2 CPU cores, 518 s (Python: 149 s). Quality control kept 286 of 300 SNPs (the 9 planted defects and 5 SNPs with MAF < 5% were removed); the levels contain 56, 32 and 16 SNPs. Mean test AUC ± SD over 50 cross-validated fits (validation cohort, 300 subjects):

| Model | Level 1 (56 SNPs) | Level 2 (32 SNPs) | Level 3 (16 SNPs) |
| :---- | :---------------: | :---------------: | :---------------: |
| Φ_I | 0.7663 ± 0.0112 | 0.7826 ± 0.0064 | 0.7767 ± 0.0061 |
| Φ_P | 0.7670 ± 0.0089 | 0.7798 ± 0.0070 | 0.7756 ± 0.0075 |
| Φ_G | 0.7646 ± 0.0134 | 0.7816 ± 0.0079 | 0.7760 ± 0.0069 |
| Φ_I+P | 0.7664 ± 0.0110 | 0.7813 ± 0.0069 | 0.7765 ± 0.0065 |
| Φ_I+G | 0.7661 ± 0.0113 | 0.7819 ± 0.0074 | 0.7764 ± 0.0069 |
| Φ_P+G | 0.7644 ± 0.0128 | 0.7807 ± 0.0078 | 0.7759 ± 0.0072 |
| NetPRS (I+P+G) | 0.7662 ± 0.0113 | 0.7813 ± 0.0071 | 0.7766 ± 0.0067 |
| wPRS (single score) | 0.7525 | 0.7754 | 0.7705 |

| Level | W_P edges (density) | W_G edges (density) | Common edges | W_G threshold |
| :---: | :-----------------: | :-----------------: | :----------: | :-----------: |
| 1 | 84 (5.45%) | 113 (7.34%) | 5 | 0.04323 (elbow, shared by all levels) |
| 2 | 26 (5.24%) | 58 (11.69%) | 4 | |
| 3 | 7 (5.83%) | 20 (16.67%) | 2 | |

What these numbers show — and do not show:

- The cross-validated models reach mean AUCs 0.004–0.015 above the single wPRS score, while the seven effect sets lie within 0.003 of each other, far less than their standard deviations: on this synthetic sample the network terms add no measurable accuracy. These are results on simulated data and say nothing about the performance reported in the paper.
- The smoothness parameters shrink during training (mean μ ≈ 0.07–0.09 at the selected epochs of levels 1–2, 0.01–0.02 at level 3) and reach 0 in the final models, where Z = X and the P_X vs P_Z comparison of Fig. 4(a) becomes a tie (0 SNPs with P_Z < P_X).
- This follows from the objective of Eq. (6): the risk depends on the parameters only through β'Z = Ωᵀx with Ω = Aβ, A = θ_I I + θ_P Q_P⁻¹ + θ_G Q_G⁻¹ and 0 ≺ A ≼ I. Hence ‖β‖² = Ωᵀ A⁻² Ω ≥ ‖Ω‖² for every risk function, with equality when μ = 0: the L2 term never rewards smoothing, and positive μ can persist only through the training path (early stopping). In a level-1 fit on the sample, the data gradient of μ also turned positive within the first 25 ADAM steps, and ADAM moves μ by about the learning rate per step, so μ fell from 1 to 0 in about 100 steps. `T39_PenaltyWithoutPropagation` checks the inequality; without the μ ≥ 0 projection the penalty would push μ below zero.

## Reproducibility and agreement with Python

- `GenerateSyntheticData` + `WriteNetPRSCSV` reproduce `dataset/sample.csv` byte for byte (SHA-256 `112a7435…a8`, test T35), as does `python make_sample_dataset.py`.
- With the same CSV file, Octave 8.4 and Python 3.13 (NumPy 2.5.3, SciPy 1.18.1) produced the same selected epoch for all 1,050 cross-validated fits and the same QC, levels, network edges, key SNPs and ranks; real-valued outputs differed by at most 1E−12 (absolute). T36 checks values of a small analysis computed by the Python version.
- The portable generator removes the dependence on platform random streams; eigenvector signs and summation order may still differ between BLAS/LAPACK libraries, which changes results only at rounding level.

## Using your own data

**CSV (recommended).** Write the genotypes (0/1/2 after QC and LD pruning), diagnosis, cohort membership, SNP-gene relations and gene-gene interactions in the format of [dataset/README.md](../dataset/README.md), set `Config.CSVFile`, choose `Config.LevelThresholds` (paper: `[3 4 5]` for ADNI) and set `Config.RunQC = false` if PLINK already filtered the SNPs.

**PLINK.** Quality control and LD pruning with the criteria of the paper, then `Config.DataSource = 'plink'`:

```bash
plink --bfile raw --geno 0.01 --hwe 1e-6 --maf 0.05 --make-bed --out qc
plink --bfile qc --indep-pairwise 50 5 0.3 --out prune
plink --bfile qc --extract prune.prune.in --make-bed --out qc_pruned
```

```matlab
Config.DataSource = 'plink';
Config.PlinkPrefix = 'qc_pruned';            % .bed/.bim/.fam
Config.ValidationIIDFile = 'validation.txt'; % IIDs of the validation cohort
Config.SNPGeneFile = 'snp_gene.txt';         % 'rsID GENE' per line (dbSNP)
Config.GGIFile = 'ggi.txt';                  % 'GENE1 GENE2 SCORE' per line (STRING)
Config.RunQC = false;
```

Optional PLINK reports of the discovery cohort can replace the internal tests: `Config.GWASFile` (`--logistic --keep-allele-order`; effect signs are aligned to the counted allele by `MatchPlinkAssoc`) and `Config.EpistasisFile` (`--epistasis --epi1 0.05`).

## Implementation notes

**Exact and fast evaluation.** Because Q = I + μL is symmetric, β'F = (Q⁻¹β)ᵀX, so β'Z = ΩᵀX with Ω = θ_Iβ + θ_P Q_P⁻¹β + θ_G Q_G⁻¹β. With one eigendecomposition of every Laplacian, risk and analytic gradients cost O(s² + sn) per epoch instead of O(s²n); `Solver = 'chol'` refactorizes Q every epoch and agrees to 1E−12.

**Settings not reported in the paper** (declared choices): Glorot-uniform β; δ = 0.005; 500 ADAM steps with the epoch of minimum validation loss; 10 × stratified 5-fold CV; per-SNP centring of X with training-fold means (the risk has no intercept); projection μ ≥ 0; σ = 1 in T_G; elbow = maximum distance to the chord of the sorted W_G values; pooled two-sample *t*-test for Fig. 4(a); exact linear SHAP of β'Z with the discovery cohort as background; mean imputation of missing genotypes with discovery-cohort means; final model trained on the whole discovery cohort for the median selected epoch (halves rounded away from zero).

**Verification.** `Test/RunAllTests.m` compares every numerical component with an independent reference: literal trace formulas of Sec. III-D with explicit inverses, central finite differences for all seven effect sets and both solvers, Eq. (8) and Eq. (10) by finite differences, the optimality of Eq. (3) for Eq. (2), plain Newton–Raphson logistic regression, the Wigginton recurrence (HWE), numerical integration of the *t* density, the theta-function form of the Kolmogorov distribution, brute-force AUC, an independent PLINK `.bed` encoder, known answers of the random generator and golden values of the Python version. Statistical sanity checks (decreasing objective, recovery of a planted network effect) are marked as such.

## Function reference

| Group | Functions |
| :---- | :-------- |
| Pipeline | `RunNetPRS`, `NetPRSDefaults`, `ExportResults`, `LoadNetPRSData` |
| Data I/O | `ReadNetPRSCSV`, `WriteNetPRSCSV`, `CSVRows`, `ReadPlinkBed`, `ReadPlinkBim`, `ReadPlinkFam`, `ReadPlinkRaw`, `ReadPlinkAssoc`, `MatchPlinkAssoc`, `ReadPlinkEpistasis`, `ReadSNPGeneRelation`, `ReadEdgeList` |
| Sample data, random streams | `GenerateSyntheticData`, `RngStream`, `RngUniform`, `RngPermutation`, `RngXorshift64` |
| Stage 1 | `SNPQualityControl`, `HWExactTest`, `ImputeGenotype`, `GWASLogistic`, `LogisticBatch`, `SelectSNPLevel`, `EpistasisTest`, `PhenotypicNetwork`, `SNPGeneMatrix`, `GenomicNetwork`, `ElbowThreshold`, `GraphLaplacian`, `NetworkSummary`, `KSTest2Asymptotic` |
| Stage 2 | `ModelInitialize`, `DataIndexing`, `ParamInitialize`, `AdamInitialize`, `ParamReshape`, `PropagationSolve`, `ForwardPropagate`, `LossCalculation`, `BackwardPropagate`, `ParameterUpdate`, `ParamTraining`, `TrainNetPRS`, `RiskPredict`, `EffectExtraction` |
| Evaluation | `RunCrossValidation`, `SummarizeCV`, `ComputeAUC`, `WeightedPRS`, `EffectSignificance`, `TwoSampleTTest`, `LinearSHAP` |
| Utilities | `SetDefaultField`, `SetRandomSeed` (test data only) |
