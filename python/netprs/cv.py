"""Repeated cross-validation of NetPRS (``RunCrossValidation``, ``SummarizeCV``)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .model import NetPRSDesign, TrainResult, train_netprs
from .stats import compute_auc

__all__ = ["run_cross_validation", "CVSummary", "summarize_cv", "mean_or_nan", "std_or_nan"]


def _train_one(args):
    design, idx_model = args
    return train_netprs(design, idx_model)[0]


def run_cross_validation(design: NetPRSDesign, num_workers: int = 0) -> List[TrainResult]:
    """Train every model of ``design.cv_list``.

    ``num_workers > 0`` trains the models in parallel processes. Each model is
    independent and seeded by its iteration, so the results do not depend on
    ``num_workers``.
    """
    n = design.num_model
    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as pool:
            results = list(pool.map(_train_one, [(design, k) for k in range(n)]))
    else:
        results = [train_netprs(design, k)[0] for k in range(n)]
    if design.params.verbose:
        for k, r in enumerate(results):
            print(f"  model {k + 1:3d}/{n} (iter {r.idx_iter}): best epoch {r.best_epoch:4d} | "
                  f"valid AUC {r.auc_valid:.4f} | test AUC {r.auc_test:.4f}")
    return results


def mean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else float("nan")


def std_or_nan(x: np.ndarray) -> float:
    """Sample standard deviation (N - 1) as MATLAB ``std``; 0 for one value."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


@dataclass
class CVSummary:
    """Aggregated cross-validation results of one model type.

    Headline performance: ``auc_mean``/``auc_sd`` of the per-model test AUCs
    ("average of AUCs"). Per-model arrays follow the rows of ``cv_list``.
    Ensemble AUCs average the risk of several models and are not comparable with
    the AUC of a single score such as wPRS.
    """

    label: str
    effect_code: str
    cv_mode: str
    num_model: int
    iteration: np.ndarray
    fold_valid: np.ndarray
    fold_test: np.ndarray           # NaN in the external mode
    best_epoch: np.ndarray
    auc_train: np.ndarray
    auc_valid: np.ndarray
    auc_test: np.ndarray
    auc_mean: float
    auc_sd: float
    theta_model: np.ndarray         # (E, num_model)
    theta: np.ndarray
    mu_model: np.ndarray            # (K, num_model)
    mu: np.ndarray
    median_best_epoch: float
    ensemble_prob: np.ndarray
    ensemble_auc: float
    auc_iter_ensemble: Optional[np.ndarray] = None   # external mode
    auc_iter_oof: Optional[np.ndarray] = None        # nested mode


def summarize_cv(design: NetPRSDesign, results: List[TrainResult], label: str = "",
                 verbose: Optional[bool] = None) -> CVSummary:
    """Aggregate cross-validated results (prints one line when ``verbose``)."""
    num_model = len(results)
    num_iter = design.params.num_iter
    iters = np.array([r.idx_iter for r in results])
    auc_test = np.array([r.auc_test for r in results])
    ok = ~np.isnan(auc_test)
    theta_model = np.column_stack([r.theta for r in results])
    mu_model = (np.column_stack([r.mu for r in results]) if design.num_net > 0
                else np.zeros((0, num_model)))
    if design.cv_mode == "external":
        y_test = design.y_all[design.num_disc:]
        prob = np.zeros(design.num_ext)
        auc_iter = np.full(num_iter, np.nan)
        for it in range(1, num_iter + 1):
            sel = np.flatnonzero(iters == it)
            if sel.size == 0:
                continue
            p_it = np.zeros(design.num_ext)
            for k in sel:
                p_it = p_it + results[k].prob_test
            auc_iter[it - 1] = compute_auc(p_it / sel.size, y_test)
            prob = prob + p_it
        ensemble_prob = prob / num_model
        ensemble_auc = compute_auc(ensemble_prob, y_test)
        extra = dict(auc_iter_ensemble=auc_iter)
    elif design.cv_mode == "nested":
        y_disc = design.y_all[:design.num_disc]
        auc_iter = np.full(num_iter, np.nan)
        acc_all = np.zeros(design.num_disc)
        cnt_all = np.zeros(design.num_disc)
        for it in range(1, num_iter + 1):
            acc = np.zeros(design.num_disc)
            cnt = np.zeros(design.num_disc)
            for k in np.flatnonzero(iters == it):
                acc[results[k].idx_test] += results[k].prob_test
                cnt[results[k].idx_test] += 1
            if np.all(cnt > 0):
                auc_iter[it - 1] = compute_auc(acc / cnt, y_disc)
            acc_all += acc
            cnt_all += cnt
        ensemble_prob = acc_all / np.maximum(cnt_all, 1)
        ensemble_auc = compute_auc(ensemble_prob, y_disc) if np.all(cnt_all > 0) else float("nan")
        extra = dict(auc_iter_oof=auc_iter)
    else:
        raise ValueError("unknown CV mode.")
    best_epoch = np.array([r.best_epoch for r in results], dtype=float)
    summary = CVSummary(
        label=label, effect_code=design.effect_code, cv_mode=design.cv_mode, num_model=num_model,
        iteration=iters, fold_valid=np.array([np.nan if r.fold_valid is None else r.fold_valid for r in results]),
        fold_test=np.array([np.nan if r.fold_test is None else r.fold_test for r in results]),
        best_epoch=best_epoch, auc_train=np.array([r.auc_train for r in results]),
        auc_valid=np.array([r.auc_valid for r in results]), auc_test=auc_test,
        auc_mean=mean_or_nan(auc_test[ok]), auc_sd=std_or_nan(auc_test[ok]),
        theta_model=theta_model, theta=theta_model.mean(axis=1), mu_model=mu_model,
        mu=mu_model.mean(axis=1) if mu_model.size else np.zeros(0),
        median_best_epoch=float(np.median(best_epoch)), ensemble_prob=ensemble_prob,
        ensemble_auc=ensemble_auc, **extra)
    if verbose is None:
        verbose = bool(label)
    if verbose:
        print(f"{label:<24s} AUC {summary.auc_mean:.4f} +/- {summary.auc_sd:.4f} ({int(ok.sum())} models; "
              f"ensemble {summary.ensemble_auc:.4f}) | theta {_vector_text(summary.theta)} | "
              f"mu {_vector_text(summary.mu)} | epoch {summary.median_best_epoch:g}")
    return summary


def _vector_text(v: np.ndarray) -> str:
    return "-" if v.size == 0 else "[" + " ".join(f"{x:.3g}" for x in v) + "]"
