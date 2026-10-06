"""Baseline and interpretation: wPRS, P_X vs P_Z (Fig. 4a) and SHAP (Fig. 4b).

Mirrors ``WeightedPRS``, ``EffectSignificance`` and ``LinearSHAP``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .stats import two_sample_ttest

__all__ = ["weighted_prs", "Significance", "effect_significance", "linear_shap"]


def weighted_prs(X: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted PRS (wPRS baseline): ``PRS_i = sum_j weight_j X[j, i]``.

    ``weight`` are GWAS log odds ratios of the discovery cohort; NaN weights
    (failed fits) contribute 0. ``X`` must be complete (impute first).
    """
    X = np.asarray(X, dtype=float)
    w = np.asarray(weight, dtype=float).ravel().copy()
    if X.shape[0] != w.size:
        raise ValueError("X must have len(weight) rows.")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN/Inf; impute missing genotypes first.")
    w[~np.isfinite(w)] = 0.0
    return w @ X


@dataclass
class Significance:
    """Two-sample t-tests of AD-positive vs AD-negative subjects per SNP."""

    p_x: np.ndarray                 # independent effect X
    p_z: np.ndarray                 # integrated effect Z
    z_more_significant: np.ndarray
    num_z_more_significant: int
    frac_z_more_significant: float
    mean_log_p_x: float             # mean -log10 P over SNPs
    mean_log_p_z: float
    relative_gain: float            # mean_log_p_z / mean_log_p_x - 1
    kind: str


def effect_significance(X: np.ndarray, Z: np.ndarray, y: np.ndarray, kind: str = "pooled") -> Significance:
    """Fig. 4(a): ``P_X`` vs ``P_Z`` for every SNP (pooled t-test unless ``kind='welch'``).

    ``P_Z < P_X`` is counted only beyond a relative 1e-9, so mathematically equal
    P-values (e.g. ``Z = X`` when every ``mu`` is 0) count as ties, never as decided by rounding.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    case = np.asarray(y).ravel().astype(bool)
    if X.shape[1] != case.size or X.shape != Z.shape:
        raise ValueError("X and Z must be s x n with n = len(y).")
    px = two_sample_ttest(X[:, case], X[:, ~case], kind)[0]
    pz = two_sample_ttest(Z[:, case], Z[:, ~case], kind)[0]
    with np.errstate(invalid="ignore"):
        more = pz < px * (1 - 1e-9)              # ties (e.g. Z = X when mu = 0) are not counted
    with np.errstate(divide="ignore"):
        lx, lz = -np.log10(px), -np.log10(pz)
    ok = np.isfinite(lx) & np.isfinite(lz)
    mx = float(lx[ok].mean()) if ok.any() else float("nan")
    mz = float(lz[ok].mean()) if ok.any() else float("nan")
    return Significance(p_x=px, p_z=pz, z_more_significant=more, num_z_more_significant=int(more.sum()),
                        frac_z_more_significant=float(more.sum() / px.size), mean_log_p_x=mx,
                        mean_log_p_z=mz, relative_gain=mz / mx - 1.0, kind=kind)


def linear_shap(beta: np.ndarray, z_explain: np.ndarray, z_background: np.ndarray) -> Tuple[np.ndarray, float]:
    """Exact SHAP values of the logit ``beta' Z`` (linear model, independent features).

    ``phi[j, i] = beta_j (Z[j, i] - mean_background Z[j])``; local accuracy:
    ``phi[:, i].sum() = beta' Z[:, i] - base`` with ``base = beta' mean_background Z``.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    z_explain = np.asarray(z_explain, dtype=float)
    z_background = np.asarray(z_background, dtype=float)
    if z_explain.shape[0] != beta.size or z_background.shape[0] != beta.size:
        raise ValueError("Z matrices must have len(beta) rows.")
    mu = z_background.mean(axis=1)
    return beta[:, None] * (z_explain - mu[:, None]), float(beta @ mu)
