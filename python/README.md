# NetPRS — Python implementation

Python version of the MATLAB code in [`../matlab`](../matlab/README.md) for **NetPRS: SNP interaction aware network-based polygenic risk score for Alzheimer's disease** (S. Park, D. Lee, J. Kim, *et al.*, *IEEE EMBS BHI 2024*, [doi:10.1109/BHI62660.2024.10913658](https://doi.org/10.1109/BHI62660.2024.10913658)).

It follows the MATLAB functions one to one, reads the same [CSV format](../dataset/README.md), uses the same portable random streams for the folds and the initial β, and writes the same result tables. On `dataset/sample.csv` it reproduces the MATLAB/Octave results (same selected epoch in all 1,050 cross-validated fits; real values within 1E−12).

## Installation

Python 3.9 or later with NumPy and SciPy (tested with Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1).

```bash
cd python
pip install -r requirements.txt          # numpy, scipy
pip install -e ".[test]"                 # optional: package + pytest, statsmodels, scikit-learn
```

The scripts also run without installation from the `python` folder.

## Quick start

```bash
python run_netprs.py                                 # complete analysis of ../dataset/sample.csv -> result/
python run_netprs.py --num-iter 2 --no-ablation      # quick run (about 10 s)
python run_netprs.py --data my_cohort.csv --thresholds 3 4 5 --no-qc --output my_result
python run_netprs.py --help
python make_sample_dataset.py --check                # verify dataset/sample.csv
python -m pytest                                     # 92 tests
```

```python
from netprs import netprs_defaults, run_netprs, export_results

config, params = netprs_defaults()           # dataset/sample.csv, paper settings
params.num_iter = 2                          # quicker run
results, info = run_netprs(config, params)   # list of LevelResult, RunInfo
export_results(results, info)                # CSV tables and run_info.json
print(results[0].netprs.auc_mean, results[0].wprs_auc)
```

Lower-level use, e.g. a NetPRS model on your own matrices:

```python
from netprs import Parameters, model_initialize, run_cross_validation, summarize_cv

design = model_initialize(X_disc, y_disc, X_val, y_val, w_phe=Wp, w_gen=Wg, params=Parameters(effects="IPG"))
summary = summarize_cv(design, run_cross_validation(design), label="NetPRS")
```

## Package layout

```text
python/
├── run_netprs.py           command line (= matlab/NetPRS.m)
├── make_sample_dataset.py  regenerate or check dataset/sample.csv
├── pyproject.toml, requirements.txt, README.md
├── netprs/                 package
├── tests/                  pytest suite (92 tests)
└── tools/compare_results.py  compare two result folders (e.g. MATLAB vs Python)
```

| Module | Content | MATLAB counterpart |
| :----- | :------ | :----------------- |
| `netprs.pipeline` | `Config`, `netprs_defaults`, `load_netprs_data`, `run_netprs`, `export_results` | `NetPRSDefaults`, `LoadNetPRSData`, `RunNetPRS`, `ExportResults` |
| `netprs.data` | `NetPRSData`, `read_netprs_csv`, `write_netprs_csv`, `edge_matrix` | `ReadNetPRSCSV`, `WriteNetPRSCSV` |
| `netprs.synthetic` | `generate_synthetic_data`, `SyntheticTruth` | `GenerateSyntheticData` |
| `netprs.rng` | `RandomStream` (uniform, permutation) | `RngStream`, `RngUniform`, `RngPermutation`, `RngXorshift64` |
| `netprs.qc` | `snp_quality_control`, `impute_genotype`, `select_snp_level` | `SNPQualityControl`, `ImputeGenotype`, `SelectSNPLevel` |
| `netprs.stats` | `logistic_batch`, `gwas_logistic`, `epistasis_test`, `hwe_exact_test`, `two_sample_ttest`, `ks_test2_asymptotic`, `compute_auc` | `LogisticBatch`, `GWASLogistic`, `EpistasisTest`, `HWExactTest`, `TwoSampleTTest`, `KSTest2Asymptotic`, `ComputeAUC` |
| `netprs.network` | `graph_laplacian`, `phenotypic_network`, `genomic_network`, `elbow_threshold`, `snp_gene_matrix`, `network_summary` | `GraphLaplacian`, `PhenotypicNetwork`, `GenomicNetwork`, `ElbowThreshold`, `SNPGeneMatrix`, `NetworkSummary` |
| `netprs.model` | `Parameters`, `model_initialize`, `NetPRSModel` (`data_indexing`, `param_initialize`, `param_reshape`, `propagation_solve`, `forward_propagate`, `loss_calculation`, `backward_propagate`, `parameter_update`, `param_training`, `risk_predict`, `effect_extraction`), `train_netprs` | `ModelInitialize`, `DataIndexing`, `ParamInitialize`, `ParamReshape`, `PropagationSolve`, `ForwardPropagate`, `LossCalculation`, `BackwardPropagate`, `ParameterUpdate`, `AdamInitialize`, `ParamTraining`, `RiskPredict`, `EffectExtraction`, `TrainNetPRS` |
| `netprs.cv` | `run_cross_validation`, `summarize_cv`, `CVSummary` | `RunCrossValidation`, `SummarizeCV` |
| `netprs.interpret` | `weighted_prs`, `effect_significance`, `linear_shap` | `WeightedPRS`, `EffectSignificance`, `LinearSHAP` |
| `netprs.plink` | PLINK `.bed/.bim/.fam/.raw`, `--logistic`, `--epistasis` readers, allele alignment, SNP-gene and edge-list readers | `ReadPlink*`, `MatchPlinkAssoc`, `ReadSNPGeneRelation`, `ReadEdgeList` |

Conventions: SNP and subject indices are 0-based in Python; iteration and fold labels (used for seeding and in the exported tables) are 1-based as in MATLAB, and `model_index` in `cv_models.csv` is 1-based. `Config` and `Parameters` are dataclasses with the MATLAB defaults in snake_case (`level_thresholds`, `max_epoch`, …); `config.verbose` prints the same progress messages as MATLAB.

## Configuration, outputs and results

The configuration fields, the seven output files and their columns, the analysis steps and the results on `sample.csv` are described in [`../matlab/README.md`](../matlab/README.md); both versions share them. In short, on `sample.csv` (default settings, 149 s on 2 CPU cores):

| Model | Level 1 (56 SNPs) | Level 2 (32 SNPs) | Level 3 (16 SNPs) |
| :---- | :---------------: | :---------------: | :---------------: |
| NetPRS (I+P+G) | 0.7662 ± 0.0113 | 0.7813 ± 0.0071 | 0.7766 ± 0.0067 |
| Φ_I | 0.7663 ± 0.0112 | 0.7826 ± 0.0064 | 0.7767 ± 0.0061 |
| wPRS | 0.7525 | 0.7754 | 0.7705 |

The smoothness parameters μ shrink to about 0.01–0.09 at the selected epochs and to 0 in the final models: under the L2 penalty of Eq. (6), Ω = Aβ with 0 ≺ A ≼ I implies ‖β‖² ≥ ‖Ω‖², so smoothing is never rewarded by the regularizer (`tests/test_model.py::test_penalty_is_smallest_without_propagation`). See the MATLAB README for the full discussion.

## Comparing with MATLAB

```bash
# after running NetPRS.m in ../matlab and run_netprs.py here
python tools/compare_results.py ../matlab/result result
```

Text and integer columns (levels, folds, selected epochs, edge counts, ranks) must be identical and real values agree within `--rtol 1e-6 --atol 1e-9` by default. For the reference runs (Octave 8.4.0 vs Python 3.13) the largest absolute difference was 1E−12.

## Tests

`python -m pytest` runs 92 tests (about 10 s) that compare the implementation with independent references: a pure-Python big-integer implementation of the random generator and known answers of the MATLAB implementation; statsmodels logistic regression; SciPy *t*- and KS-tests; scikit-learn and brute-force AUC; the Wigginton HWE recurrence; explicit inverses and the literal gradient formulas of Sec. III-D; central finite differences for all seven effect sets and both solvers; an independent PLINK `.bed` encoder; golden values of a small analysis computed by MATLAB/Octave; byte-identical reproduction of `dataset/sample.csv`; and the command-line scripts. Tests that need statsmodels or scikit-learn are skipped when these packages are missing.
