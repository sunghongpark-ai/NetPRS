from __future__ import annotations

import json
import math
import platform
import time
import tempfile
import warnings
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
import scipy.sparse as sp

from .cv import CVSummary, run_cross_validation, summarize_cv
from .data import NetPRSData, read_netprs_csv
from .interpret import Significance, effect_significance, linear_shap, weighted_prs
from .model import EFFECT_CODES, Parameters, Split, model_initialize, train_netprs
from .network import (GenomicInfo, NetworkSummary, PhenotypicInfo, genomic_network, network_summary,
                      phenotypic_network, snp_gene_matrix)
from .plink import (match_plink_assoc, read_edge_list, read_plink_assoc, read_plink_bed, read_plink_epistasis,
                    read_snp_gene_relation)
from .qc import QCResult, impute_genotype, select_snp_level, snp_quality_control
from .stats import compute_auc, epistasis_test, gwas_logistic
from .synthetic import generate_synthetic_data

__all__ = [
    "Config", "netprs_defaults", "load_netprs_data", "Interpretation", "AblationEntry", "LevelResult",
    "RunInfo", "run_netprs", "export_results", "ABLATION_NAMES",
]

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
ABLATION_NAMES = {"I": "Phi_I", "P": "Phi_P", "G": "Phi_G", "IP": "Phi_I+P", "IG": "Phi_I+G",
                  "PG": "Phi_P+G", "IPG": "NetPRS"}


@dataclass
class Config:


    data_source: str = "csv"
    csv_file: str = str(REPO_ROOT / "dataset" / "sample.csv")
    plink_prefix: str = ""
    validation_iid_file: str = ""
    snp_gene_file: str = ""
    ggi_file: str = ""
    ggi_weight_scale: float = 1.0 / 1000.0
    gwas_file: str = ""
    epistasis_file: str = ""
    synthetic_options: Dict[str, Any] = field(default_factory=dict)

    run_qc: bool = True
    qc_options: Dict[str, Any] = field(default_factory=dict)
    level_thresholds: Tuple[float, ...] = (1.0, 1.5, 2.0)
    levels: Tuple[int, ...] = (1, 2, 3)
    min_level_snp: int = 3
    epistasis_p_threshold: float = 0.05
    genomic_mu: float = 1.0
    genomic_sigma: float = 1.0
    genomic_threshold: Union[str, float] = "elbow"

    effects: str = "IPG"
    run_ablation: bool = True
    run_baseline: bool = True
    run_interpretation: bool = True
    num_key_snp: int = 10
    num_workers: int = 0

    output_dir: str = str(Path(tempfile.gettempdir()) / "NetPRS_python_results")
    verbose: bool = True

    def validate(self) -> None:
        if self.data_source not in ("csv", "plink", "synthetic"):
            raise ValueError("data_source must be 'csv', 'plink' or 'synthetic'.")
        t = np.asarray(self.level_thresholds, dtype=float).ravel()
        if t.size == 0 or np.any(~np.isfinite(t)):
            raise ValueError("level_thresholds must be finite -log10(P) cut-offs.")
        lv = list(self.levels)
        if not lv or any(int(v) != v or not 1 <= v <= t.size for v in lv) or len(set(lv)) != len(lv):
            raise ValueError(f"levels must list distinct levels in 1..{t.size}.")
        if not (int(self.min_level_snp) == self.min_level_snp and self.min_level_snp >= 2):
            raise ValueError("min_level_snp must be an integer >= 2.")
        if not 0 < self.epistasis_p_threshold <= 1:
            raise ValueError("epistasis_p_threshold must lie in (0, 1].")
        if self.effects.upper() not in EFFECT_CODES:
            raise ValueError("effects must be one of I, P, G, IP, IG, PG, IPG.")
        if not (int(self.num_key_snp) == self.num_key_snp and self.num_key_snp >= 0):
            raise ValueError("num_key_snp must be a non-negative integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")


def netprs_defaults() -> Tuple[Config, Parameters]:

    return Config(), Parameters()


def load_netprs_data(config: Config) -> NetPRSData:


    if config.data_source == "csv":
        if not config.csv_file:
            raise ValueError("csv_file is required for CSV input.")
        return read_netprs_csv(config.csv_file)
    if config.data_source == "synthetic":
        return generate_synthetic_data(**config.synthetic_options)[0]
    if config.data_source == "plink":
        for name in ("plink_prefix", "validation_iid_file", "snp_gene_file", "ggi_file"):
            if not getattr(config, name):
                raise ValueError(f"{name} is required for PLINK input.")
        X, bim, fam = read_plink_bed(config.plink_prefix)
        val_iid = set(Path(config.validation_iid_file).read_text(encoding="utf-8").split())
        is_val = np.array([x in val_iid for x in fam.iid])
        if not is_val.any():
            warnings.warn(f"No subject of {config.validation_iid_file} was found in the fileset.",
                          RuntimeWarning, stacklevel=2)
        has_y = ~np.isnan(fam.y)
        rel_snp, rel_gene = read_snp_gene_relation(config.snp_gene_file)
        scale = config.ggi_weight_scale or 1.0
        ggi_genes = read_edge_list(config.ggi_file, weight_scale=scale)[1]
        genes = sorted(set(ggi_genes) | set(rel_gene))
        ggi = read_edge_list(config.ggi_file, node_ids=genes, weight_scale=scale)[0]
        return NetPRSData(genotype=X[:, has_y], diagnosis=fam.y[has_y], is_validation=is_val[has_y],
                          subject_id=[x for x, k in zip(fam.iid, has_y) if k], snp_id=bim.snp, gene_id=genes,
                          ggi=ggi, rel_snp=rel_snp, rel_gene=rel_gene, rel_weight=np.ones(len(rel_snp)),
                          allele1=bim.a1, allele2=bim.a2, source="plink")
    raise ValueError("data_source must be 'csv', 'plink' or 'synthetic'.")


@dataclass
class Interpretation:


    max_epoch: int
    significance: Significance
    shap: np.ndarray
    shap_base: float
    mean_abs_shap: np.ndarray
    shap_case: np.ndarray
    shap_control: np.ndarray
    shap_rank: np.ndarray
    key_snp: List[str]
    beta: np.ndarray
    theta: np.ndarray
    mu: np.ndarray
    effect_code: str
    effective_weight: np.ndarray
    auc_validation: float


@dataclass
class AblationEntry:
    name: str
    effects: str
    summary: CVSummary


@dataclass
class LevelResult:
    level: int
    threshold: float
    snp: List[str]
    snp_index: np.ndarray
    w_phe: np.ndarray
    w_gen: np.ndarray
    network_summary: NetworkSummary
    netprs: CVSummary
    ablation: List[AblationEntry]
    wprs_score: Optional[np.ndarray]
    wprs_auc: float
    interpretation: Optional[Interpretation]


@dataclass
class RunInfo:
    config: Config
    params: Parameters
    data: Dict[str, Any]
    keep_qc: np.ndarray
    qc: Optional[QCResult]
    all_snp: List[str]
    snp: List[str]
    gwas: Any
    level_idx: List[np.ndarray]
    level_num_snp: np.ndarray
    analysed_levels: List[int]
    effects: str
    has_genomic: bool
    network_snp_index: np.ndarray
    w_phe_all: np.ndarray
    w_gen_all: np.ndarray
    phenotypic: PhenotypicInfo
    genomic: Optional[GenomicInfo]
    genomic_threshold: float
    software: str
    seconds: float


def _log(config: Config, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def run_netprs(config: Optional[Config] = None, params: Optional[Parameters] = None
               ) -> Tuple[List[LevelResult], RunInfo]:


    config = config or Config()
    params = params or Parameters()
    config.validate()
    params.validate()
    t_start = time.perf_counter()


    data = load_netprs_data(config)
    disc = ~data.is_validation
    y = data.diagnosis
    info_data = dict(
        Source=data.source, NumSNP=data.num_snp, NumSubject=data.num_subject, NumDiscovery=int(disc.sum()),
        NumValidation=int((~disc).sum()), NumCaseDiscovery=int((y[disc] == 1).sum()),
        NumCaseValidation=int((y[~disc] == 1).sum()), NumGene=len(data.gene_id), NumRelation=len(data.rel_snp),
        NumGGIEdge=int(sp.triu(data.ggi, k=1).nnz), NumMissing=int(np.isnan(data.genotype).sum()))
    _log(config, f"{data.source} data: {data.num_snp} SNPs, {data.num_subject} subjects (discovery "
                 f"{info_data['NumDiscovery']} with {info_data['NumCaseDiscovery']} cases / validation "
                 f"{info_data['NumValidation']} with {info_data['NumCaseValidation']} cases)")
    _log(config, f"  {info_data['NumGene']} genes, {info_data['NumRelation']} SNP-gene relations, "
                 f"{info_data['NumGGIEdge']} gene-gene interactions, {info_data['NumMissing']} missing calls")
    if np.unique(y[disc]).size < 2:
        raise ValueError("the discovery cohort must contain cases and controls.")
    if not (~disc).any():
        raise ValueError("the validation cohort is empty.")
    if np.unique(y[~disc]).size < 2:
        warnings.warn("The validation cohort lacks cases or controls; test AUCs are NaN.", RuntimeWarning, stacklevel=2)


    if config.run_qc:
        qc = snp_quality_control(data.genotype[:, disc], y[disc], **config.qc_options)
        keep = qc.keep
    else:
        qc, keep = None, np.ones(data.num_snp, dtype=bool)
    geno = impute_genotype(data.genotype[keep], np.flatnonzero(disc))[0]
    snp_id = [x for x, k in zip(data.snp_id, keep) if k]
    _log(config, f"Quality control: {len(snp_id)} of {keep.size} SNPs kept")
    if not config.gwas_file:
        gwas = gwas_logistic(geno[:, disc], y[disc])
    else:
        a1 = [x for x, k in zip(data.allele1, keep) if k] if data.allele1 else None
        a2 = [x for x, k in zip(data.allele2, keep) if k] if data.allele2 else None
        gwas = match_plink_assoc(read_plink_assoc(config.gwas_file), snp_id, a1, a2)
    level_idx, level_info = select_snp_level(gwas.p, config.level_thresholds)
    thresholds = list(np.asarray(config.level_thresholds, dtype=float).ravel())
    for l, (t, num) in enumerate(zip(thresholds, level_info.num_snp), start=1):
        _log(config, f"  level {l} (-log10 P > {t:g}): {num} SNPs")
    analysed = [int(l) for l in config.levels if level_info.num_snp[int(l) - 1] >= config.min_level_snp]
    for l in sorted(set(int(v) for v in config.levels) - set(analysed)):
        _log(config, f"  level {l} skipped (fewer than {config.min_level_snp} SNPs)")
    if not analysed:
        raise ValueError(f"no SNP level has {config.min_level_snp} or more SNPs; lower level_thresholds.")


    all_idx = np.unique(np.concatenate([level_idx[l - 1] for l in analysed]))
    y_disc = y[disc]
    if not config.epistasis_file:
        epi = epistasis_test(geno[np.ix_(all_idx, np.flatnonzero(disc))], y_disc)
    else:
        epi = read_plink_epistasis(config.epistasis_file, [snp_id[j] for j in all_idx])
    w_phe_all, info_p = phenotypic_network(epi.or_int, epi.p, p_threshold=config.epistasis_p_threshold)
    has_genomic = len(data.rel_snp) > 0
    effects = config.effects.upper()
    info_g: Optional[GenomicInfo] = None
    if has_genomic:
        G = snp_gene_matrix([snp_id[j] for j in all_idx], data.gene_id, data.rel_snp, data.rel_gene,
                            data.rel_weight)
        w_gen_all, info_g = genomic_network(G, data.ggi, mu=config.genomic_mu, sigma=config.genomic_sigma,
                                            threshold=config.genomic_threshold)
        g_threshold, g_edges = info_g.threshold, info_g.num_edge
    else:
        warnings.warn("No SNP-gene relations: W_G cannot be built and effects with G are skipped.",
                      RuntimeWarning, stacklevel=2)
        w_gen_all = np.zeros((all_idx.size, all_idx.size))
        g_threshold, g_edges = float("nan"), 0
        effects = effects.replace("G", "")
        if not effects:
            raise ValueError("effects needs I or P when W_G is unavailable.")
    _log(config, f"Networks on {all_idx.size} SNPs: W_P {info_p.num_edge} edges "
                 f"(P < {config.epistasis_p_threshold:g}), W_G {g_edges} edges (threshold {g_threshold:.6g})")


    results = [
        _analyse_level(level, level_idx[level - 1], all_idx, w_phe_all, w_gen_all, geno, y, disc, snp_id,
                       gwas.beta, effects, has_genomic, thresholds[level - 1], config, params)
        for level in analysed]

    _print_summary(config, results)
    info = RunInfo(
        config=config, params=params, data=info_data, keep_qc=keep, qc=qc, all_snp=list(data.snp_id),
        snp=snp_id, gwas=gwas, level_idx=level_idx, level_num_snp=level_info.num_snp, analysed_levels=analysed,
        effects=effects, has_genomic=has_genomic, network_snp_index=all_idx, w_phe_all=w_phe_all,
        w_gen_all=w_gen_all, phenotypic=info_p, genomic=info_g, genomic_threshold=g_threshold,
        software=f"Python {platform.python_version()} (NumPy {np.__version__}, SciPy {scipy.__version__})",
        seconds=time.perf_counter() - t_start)
    _log(config, f"Finished in {info.seconds:.1f} s")
    return results, info


def _analyse_level(level: int, idx: np.ndarray, all_idx: np.ndarray, w_phe_all: np.ndarray,
                   w_gen_all: np.ndarray, geno: np.ndarray, y: np.ndarray, disc: np.ndarray, snp_id: List[str],
                   gwas_beta: np.ndarray, effects: str, has_genomic: bool, threshold: float, config: Config,
                   params: Parameters) -> LevelResult:
    pos = np.flatnonzero(np.isin(all_idx, idx))
    w_phe = w_phe_all[np.ix_(pos, pos)]
    w_gen = w_gen_all[np.ix_(pos, pos)]
    x_disc, x_val = geno[np.ix_(idx, np.flatnonzero(disc))], geno[np.ix_(idx, np.flatnonzero(~disc))]
    y_disc, y_val = y[disc], y[~disc]
    snp = [snp_id[j] for j in idx]
    _log(config, f"\n================ Level {level}: {idx.size} SNPs (-log10 P > {threshold:g}) ================")
    net_sum = network_summary(w_phe, w_gen)
    if config.verbose:
        print(f"{'Network (Table III)':<28s} {'Nodes':>6s} | {'Edges_P':>8s} {'Avg_P':>7s} {'Dens_P':>9s} | "
              f"{'Edges_G':>8s} {'Avg_G':>7s} {'Dens_G':>9s} | {'Common':>6s} {'Corr':>8s} {'KS P':>10s}")
        n_ = net_sum
        print(f"{'Level ' + str(level):<28s} {n_.num_node:6d} | {n_.phe.num_edge:8d} {n_.phe.avg_edge:7.4f} "
              f"{100 * n_.phe.density:8.4f}% | {n_.gen.num_edge:8d} {n_.gen.avg_edge:7.4f} "
              f"{100 * n_.gen.density:8.4f}% | {n_.num_common:6d} {n_.corr:8.4f} {n_.ks_p:10.2e}")

    def design_for(code: str):
        p = replace(params, effects=code)
        return model_initialize(x_disc, y_disc, x_val, y_val, w_phe=w_phe, w_gen=w_gen, snp=snp, params=p)


    main_design = design_for(effects)
    summary_main = summarize_cv(main_design, run_cross_validation(main_design, config.num_workers),
                                f"{ABLATION_NAMES[effects]} ({effects})", config.verbose)

    ablation: List[AblationEntry] = []
    if config.run_ablation:
        for code in EFFECT_CODES:
            if not has_genomic and "G" in code:
                continue
            if code == effects:
                summary = summary_main
            else:
                d = design_for(code)
                summary = summarize_cv(d, run_cross_validation(d, config.num_workers),
                                       f"{ABLATION_NAMES[code]} ({code})", config.verbose)
            ablation.append(AblationEntry(name=ABLATION_NAMES[code], effects=code, summary=summary))

    wprs_score, wprs_auc = None, float("nan")
    if config.run_baseline:
        wprs_score = weighted_prs(x_val, gwas_beta[idx])
        wprs_auc = compute_auc(wprs_score, y_val)
        _log(config, f"{'wPRS':<24s} AUC {wprs_auc:.4f} (single score)")

    interp = _interpretation(main_design, summary_main, x_disc, x_val, y_val, snp, config) \
        if config.run_interpretation else None
    return LevelResult(level=level, threshold=threshold, snp=snp, snp_index=idx, w_phe=w_phe, w_gen=w_gen,
                       network_summary=net_sum, netprs=summary_main, ablation=ablation, wprs_score=wprs_score,
                       wprs_auc=wprs_auc, interpretation=interp)


def _round_half_away(x: float) -> int:

    return int(math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1))


def _interpretation(design, summary: CVSummary, x_disc, x_val, y_val, snp, config) -> Interpretation:
    max_epoch = _round_half_away(summary.median_best_epoch)
    final_design = replace(design, params=replace(design.params, max_epoch=max_epoch))
    split = Split(idx_train=np.arange(design.num_disc), idx_valid=(),
                  idx_test=design.num_disc + np.arange(design.num_ext), idx_iter=1)
    _, final = train_netprs(final_design, split)
    eff_val = final.effect_extraction(x_val)
    eff_disc = final.effect_extraction(x_disc)
    sig = effect_significance(x_val, eff_val.Z, y_val)
    shap, base = linear_shap(final.beta, eff_val.Z, eff_disc.Z)
    mean_abs = np.abs(shap).mean(axis=1)
    order = np.argsort(-mean_abs, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(1, order.size + 1)
    top = order[:min(config.num_key_snp, order.size)]
    case = y_val == 1
    interp = Interpretation(
        max_epoch=max_epoch, significance=sig, shap=shap, shap_base=base, mean_abs_shap=mean_abs,
        shap_case=shap[:, case].mean(axis=1), shap_control=shap[:, ~case].mean(axis=1), shap_rank=rank,
        key_snp=[snp[j] for j in top], beta=final.beta.copy(), theta=final.theta.copy(), mu=final.mu.copy(),
        effect_code=design.effect_code, effective_weight=final.omega.copy(),
        auc_validation=compute_auc(final.risk_predict(x_val)[0], y_val))
    _log(config, f"Final model ({max_epoch} epochs): theta [{' '.join(f'{v:.3g}' for v in final.theta)}] | "
                 f"mu [{' '.join(f'{v:.3g}' for v in final.mu)}] | validation AUC {interp.auc_validation:.4f}")
    _log(config, f"P_Z < P_X for {sig.num_z_more_significant} / {sig.p_x.size} SNPs "
                 f"({100 * sig.frac_z_more_significant:.1f}%); mean -log10 P: X {sig.mean_log_p_x:.3f}, "
                 f"Z {sig.mean_log_p_z:.3f} ({100 * sig.relative_gain:+.1f}%)")
    _log(config, "Key SNPs by mean |SHAP| (mean SHAP in cases / controls):")
    for j in top:
        _log(config, f"  {snp[j]:<12s} {interp.shap_case[j]:+7.3f} / {interp.shap_control[j]:+7.3f}")
    return interp


def _print_summary(config: Config, results: List[LevelResult]) -> None:
    if not config.verbose or not results or not results[0].ablation:
        return
    print("\nMean test AUC of the cross-validated models (validation cohort)")
    print(f"{'Model':<10s}" + "".join(f"   Level {r.level}" for r in results))
    for a, entry in enumerate(results[0].ablation):
        print(f"{entry.name.replace('Phi_', ''):<10s}"
              + "".join(f"   {r.ablation[a].summary.auc_mean:.4f} " for r in results))
    if config.run_baseline:
        print(f"{'wPRS':<10s}" + "".join(f"   {r.wprs_auc:.4f} " for r in results))


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    x = float(v)
    if math.isnan(x):
        return ""
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.10g}"


def _write_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(_fmt(v) for v in row) + "\n")
    return path


def _effect_columns(code: str, theta: np.ndarray, mu: np.ndarray) -> List[float]:
    th = [float("nan")] * 3
    for value, letter in zip(theta, [c for c in "IPG" if c in code]):
        th["IPG".index(letter)] = float(value)
    m = [float("nan")] * 2
    for value, letter in zip(mu, [c for c in "PG" if c in code]):
        m["PG".index(letter)] = float(value)
    return th + m


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in (obj.tolist() if isinstance(obj, np.ndarray) else obj)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


def export_results(results: List[LevelResult], info: RunInfo, output_dir: Optional[Union[str, Path]] = None
                   ) -> List[Path]:


    out = Path(output_dir or info.config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []

    summary_rows, model_rows = [], []
    for r in results:
        entries = r.ablation or [AblationEntry("NetPRS", r.netprs.effect_code, r.netprs)]
        for e in entries:
            s = e.summary
            summary_rows.append([r.level, r.threshold, len(r.snp), e.name, s.effect_code, s.auc_mean, s.auc_sd,
                                 s.ensemble_auc, s.num_model, s.median_best_epoch,
                                 *_effect_columns(s.effect_code, s.theta, s.mu)])
            for k in range(s.num_model):
                mu_k = s.mu_model[:, k] if s.mu_model.size else np.zeros(0)
                model_rows.append([r.level, e.name, s.effect_code, k + 1, s.iteration[k], s.fold_valid[k],
                                   s.fold_test[k], s.best_epoch[k], s.auc_train[k], s.auc_valid[k], s.auc_test[k],
                                   *_effect_columns(s.effect_code, s.theta_model[:, k], mu_k)])
        if info.config.run_baseline:
            summary_rows.append([r.level, r.threshold, len(r.snp), "wPRS", "", r.wprs_auc] + [float("nan")] * 9)
    files.append(_write_table(out / "auc_summary.csv",
                              ["level", "threshold", "num_snp", "model", "effects", "auc_mean", "auc_sd",
                               "ensemble_auc", "num_models", "median_best_epoch", "theta_I", "theta_P", "theta_G",
                               "mu_P", "mu_G"], summary_rows))
    files.append(_write_table(out / "cv_models.csv",
                              ["level", "model", "effects", "model_index", "iteration", "valid_fold", "test_fold",
                               "best_epoch", "auc_train", "auc_valid", "auc_test", "theta_I", "theta_P", "theta_G",
                               "mu_P", "mu_G"], model_rows))

    net_rows = []
    for r in results:
        t = r.network_summary
        net_rows.append([r.level, t.num_node, t.phe.num_edge, t.phe.avg_edge, t.phe.density, t.gen.num_edge,
                         t.gen.avg_edge, t.gen.density, t.num_common, t.corr, t.ks_p])
    files.append(_write_table(out / "networks.csv",
                              ["level", "num_snp", "edges_P", "avg_weight_P", "density_P", "edges_G",
                               "avg_weight_G", "density_G", "common_edges", "corr_common", "ks_p_common"], net_rows))

    num_all = len(info.all_snp)
    qc_val = np.full((num_all, 3), np.nan)
    if info.qc is not None:
        qc_val = np.column_stack([info.qc.call_rate, info.qc.maf, info.qc.hwe_p])
    kept = np.flatnonzero(info.keep_qc)
    gw = np.full((num_all, 3), np.nan)
    se = getattr(info.gwas, "se", np.full(kept.size, np.nan))
    gw[kept] = np.column_stack([info.gwas.beta, se, info.gwas.p])
    level = np.zeros(num_all, dtype=np.int64)
    for l, idx in enumerate(info.level_idx, start=1):
        level[kept[idx]] = l
    with np.errstate(divide="ignore", invalid="ignore"):
        neglog = -np.log10(gw[:, 2])
    snp_rows = [[info.all_snp[j], *qc_val[j], int(info.keep_qc[j]), *gw[j], neglog[j], level[j]]
                for j in range(num_all)]
    files.append(_write_table(out / "snps.csv",
                              ["snp", "qc_call_rate", "qc_maf", "qc_hwe_p", "qc_pass", "gwas_beta", "gwas_se",
                               "gwas_p", "gwas_neglog10p", "level"], snp_rows))

    interp_rows, final_rows = [], []
    for r in results:
        x = r.interpretation
        if x is None:
            continue
        g = x.significance
        for j, name in enumerate(r.snp):
            interp_rows.append([r.level, name, x.beta[j], x.effective_weight[j], g.p_x[j], g.p_z[j],
                                x.mean_abs_shap[j], x.shap_case[j], x.shap_control[j], x.shap_rank[j]])
        final_rows.append([r.level, x.effect_code, x.max_epoch, *_effect_columns(x.effect_code, x.theta, x.mu),
                           x.auc_validation, g.num_z_more_significant, g.frac_z_more_significant, g.mean_log_p_x,
                           g.mean_log_p_z, g.relative_gain, x.shap_base])
    files.append(_write_table(out / "interpretation.csv",
                              ["level", "snp", "beta", "effective_weight", "p_x", "p_z", "mean_abs_shap",
                               "shap_case", "shap_control", "shap_rank"], interp_rows))
    files.append(_write_table(out / "final_model.csv",
                              ["level", "effects", "max_epoch", "theta_I", "theta_P", "theta_G", "mu_P", "mu_G",
                               "auc_validation", "num_z_more_significant", "frac_z_more_significant",
                               "mean_neglog10_p_x", "mean_neglog10_p_z", "relative_gain", "shap_base"], final_rows))

    run = dict(software=info.software, created=datetime.now().isoformat(timespec="seconds"), seconds=info.seconds,
               data=info.data, num_snp_qc=len(info.snp), level_num_snp=info.level_num_snp,
               analysed_levels=info.analysed_levels, effects=info.effects, has_genomic=info.has_genomic,
               genomic_threshold=info.genomic_threshold, config=info.config, parameter=info.params)
    path = out / "run_info.json"
    path.write_text(json.dumps(_jsonable(run), indent=1) + "\n", encoding="utf-8")
    files.append(path)
    return files
