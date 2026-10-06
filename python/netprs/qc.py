"""SNP screening: quality control, mean imputation and SNP levels (Sec. IV-B).

Mirrors ``SNPQualityControl``, ``ImputeGenotype`` and ``SelectSNPLevel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .stats import hwe_exact_test

__all__ = ["QCResult", "snp_quality_control", "impute_genotype", "LevelInfo", "select_snp_level"]


@dataclass
class QCResult:
    """Per-SNP quality-control statistics and pass flags."""

    call_rate: np.ndarray
    maf: np.ndarray
    hwe_p: np.ndarray
    pass_call: np.ndarray
    pass_maf: np.ndarray
    pass_hwe: np.ndarray
    keep: np.ndarray


def snp_quality_control(X: np.ndarray, y: np.ndarray, min_call_rate: float = 0.99, min_maf: float = 0.05,
                        min_hwe_p: float = 1e-6, hwe_samples: str = "controls") -> QCResult:
    """SNP quality control with the criteria of the paper and PLINK 1.9 semantics.

    Keeps SNPs with missing rate ``<= 1 - min_call_rate`` (``--geno 0.01``),
    ``MAF >= min_maf`` (``--maf 0.05``) and HWE exact-test ``P >= min_hwe_p``
    (``--hwe 1e-6``, controls only by default). LD pruning should be run in PLINK.
    ``X`` is (s, n) with hard calls 0/1/2 or NaN; ``y`` is (n,) with 1 = case.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    s, n = X.shape
    if y.size != n:
        raise ValueError("y must have one entry per column of X.")
    obs = np.isfinite(X)
    if np.any(obs & ((X != np.round(X)) | (X < 0) | (X > 2))):
        raise ValueError("X must contain hard genotype calls 0/1/2 (or NaN).")
    call_rate = obs.sum(axis=1) / n
    Xz = np.where(obs, X, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        af = Xz.sum(axis=1) / (2 * obs.sum(axis=1))
    maf = np.minimum(af, 1 - af)
    if hwe_samples == "controls":
        cols = y == 0
    elif hwe_samples == "all":
        cols = np.ones(n, dtype=bool)
    else:
        raise ValueError("hwe_samples must be 'controls' or 'all'.")
    Xh = X[:, cols]
    hwe_p = hwe_exact_test((Xh == 0).sum(axis=1), (Xh == 1).sum(axis=1), (Xh == 2).sum(axis=1))
    pass_call = (1 - call_rate) <= (1 - min_call_rate) + 1e-12
    with np.errstate(invalid="ignore"):
        pass_maf = maf >= min_maf
        pass_hwe = ~(hwe_p < min_hwe_p) & ~np.isnan(hwe_p)
    return QCResult(call_rate=call_rate, maf=maf, hwe_p=hwe_p, pass_call=pass_call, pass_maf=pass_maf,
                    pass_hwe=pass_hwe, keep=pass_call & pass_maf & pass_hwe)


def impute_genotype(X: np.ndarray, ref_cols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean imputation: NaN entries of SNP j get the mean dosage of SNP j over
    the reference subjects ``ref_cols`` (index or bool array; default all).

    Use the discovery cohort as reference so that the validation cohort does not
    inform the imputation. Returns ``(X_imputed, means)``.
    """
    X = np.array(X, dtype=float)
    R = X if ref_cols is None else X[:, ref_cols]
    obs = np.isfinite(R)
    cnt = obs.sum(axis=1)
    mean = np.where(obs, R, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    missing = ~np.isfinite(X)
    if np.any((cnt == 0) & missing.any(axis=1)):
        raise ValueError("a SNP with missing values has no genotyped reference subject.")
    rows, cols = np.nonzero(missing)
    X[rows, cols] = mean[rows]
    return X, mean


@dataclass
class LevelInfo:
    num_snp: np.ndarray
    thresholds: np.ndarray
    fraction: np.ndarray


def select_snp_level(p: np.ndarray, thresholds: Sequence[float]) -> Tuple[List[np.ndarray], LevelInfo]:
    """SNP levels: SNP j belongs to level l when ``-log10(P_j) > thresholds[l]``.

    Returns the 0-based SNP indices of each level (in SNP order) and a summary.
    The paper used ``[3, 4, 5]`` (ADNI) and ``[4, 5, 6]`` (BICWALZS).
    """
    p = np.asarray(p, dtype=float).ravel()
    finite = np.isfinite(p)
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("P-values must lie in [0, 1].")
    with np.errstate(divide="ignore", invalid="ignore"):
        score = -np.log10(p)
        idx = [np.flatnonzero(score > t) for t in thresholds]
    num = np.array([i.size for i in idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = num / finite.sum()
    return idx, LevelInfo(num_snp=num, thresholds=np.asarray(thresholds, dtype=float), fraction=fraction)
