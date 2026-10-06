"""Readers of PLINK 1.9 files and of the biological-knowledge text files.

Mirrors ``ReadPlinkBed``/``ReadPlinkBim``/``ReadPlinkFam``/``ReadPlinkRaw``,
``ReadPlinkAssoc``/``MatchPlinkAssoc``/``ReadPlinkEpistasis``,
``ReadSNPGeneRelation`` and ``ReadEdgeList``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .data import edge_matrix
from .stats import pairs_to_matrix

__all__ = [
    "Bim", "Fam", "read_plink_bim", "read_plink_fam", "read_plink_bed", "Raw", "read_plink_raw", "Assoc",
    "read_plink_assoc", "MatchedAssoc", "match_plink_assoc", "PlinkEpistasis", "read_plink_epistasis",
    "read_snp_gene_relation", "read_edge_list",
]

PathLike = Union[str, Path]


def _rows(path: PathLike) -> List[List[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.split() for line in handle if line.strip()]


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _case_control(pheno: np.ndarray) -> np.ndarray:
    y = np.full(pheno.shape, np.nan)
    y[pheno == 2] = 1.0
    y[pheno == 1] = 0.0
    return y


@dataclass
class Bim:
    chr: List[str]
    snp: List[str]
    cm: np.ndarray
    bp: np.ndarray
    a1: List[str]                   # counted allele (dosage of read_plink_bed)
    a2: List[str]

    def subset(self, index: np.ndarray) -> "Bim":
        pick = lambda v: [v[i] for i in index]          # noqa: E731
        return Bim(pick(self.chr), pick(self.snp), self.cm[index], self.bp[index], pick(self.a1), pick(self.a2))


@dataclass
class Fam:
    fid: List[str]
    iid: List[str]
    pat: List[str]
    mat: List[str]
    sex: np.ndarray
    pheno: np.ndarray
    y: np.ndarray                   # 2 -> 1 (case), 1 -> 0 (control), otherwise NaN


def read_plink_bim(path: PathLike) -> Bim:
    rows = _rows(path)
    if any(len(r) < 6 for r in rows):
        raise ValueError(f"{path}: every line needs 6 fields.")
    bim = Bim([r[0] for r in rows], [r[1] for r in rows], np.array([_float(r[2]) for r in rows]),
              np.array([_float(r[3]) for r in rows]), [r[4] for r in rows], [r[5] for r in rows])
    if len(set(bim.snp)) != len(bim.snp):
        warnings.warn(f"Duplicated variant IDs in {path}.", RuntimeWarning, stacklevel=2)
    return bim


def read_plink_fam(path: PathLike) -> Fam:
    rows = _rows(path)
    if any(len(r) < 6 for r in rows):
        raise ValueError(f"{path}: every line needs 6 fields.")
    pheno = np.array([_float(r[5]) for r in rows])
    return Fam([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
               np.array([_float(r[4]) for r in rows]), pheno, _case_control(pheno))


def read_plink_bed(prefix: PathLike, snp_index: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Bim, Fam]:
    """Genotypes of a PLINK 1 binary fileset (``.bed/.bim/.fam``).

    ``X[j, i]`` is the dosage of the first ``.bim`` allele (A1) of variant
    ``snp_index[j]`` in sample i: code 00 -> 2, 10 -> 1, 11 -> 0, 01 -> NaN
    (SNP-major mode; the low-order bit pair holds the first sample of a byte).
    ``snp_index``: 0-based indices or a bool mask (default all variants).
    """
    prefix = str(prefix)
    bim = read_plink_bim(prefix + ".bim")
    fam = read_plink_fam(prefix + ".fam")
    num_var, n = len(bim.snp), len(fam.iid)
    if snp_index is None:
        index = np.arange(num_var)
    else:
        index = np.asarray(snp_index)
        if index.dtype == bool:
            if index.size != num_var:
                raise ValueError("a bool snp_index needs one entry per variant.")
            index = np.flatnonzero(index)
        index = index.astype(np.int64).ravel()
        if np.any((index < 0) | (index >= num_var)):
            raise ValueError("snp_index out of range.")
    nb = (n + 3) // 4
    with open(prefix + ".bed", "rb") as handle:
        magic = handle.read(3)
        if len(magic) < 3 or magic[0] != 0x6C or magic[1] != 0x1B:
            raise ValueError("not a PLINK 1 .bed file (wrong magic number).")
        if magic[2] != 1:
            raise ValueError("only SNP-major .bed files are supported.")
        raw = np.frombuffer(handle.read(), dtype=np.uint8)
    if raw.size < nb * num_var:
        raise ValueError("the .bed file is truncated.")
    raw = raw[:nb * num_var].reshape(num_var, nb)[index]           # (k, nb)
    code2dose = np.array([2.0, np.nan, 1.0, 0.0])
    lut = np.stack([code2dose[(np.arange(256) >> (2 * slot)) & 3] for slot in range(4)], axis=1)  # (256, 4)
    X = lut[raw].reshape(index.size, 4 * nb)[:, :n]
    return X, bim.subset(index), fam


@dataclass
class Raw:
    X: np.ndarray                   # (s, n) dosage of the counted allele (NaN = NA)
    snp: List[str]
    counted_allele: List[str]
    fid: List[str]
    iid: List[str]
    sex: np.ndarray
    pheno: np.ndarray
    y: np.ndarray


def read_plink_raw(path: PathLike) -> Raw:
    """PLINK additive genotype file (``--recode A``, ``.raw``)."""
    rows = _rows(path)
    if not rows:
        raise ValueError(f"empty file {path}.")
    header = rows[0]
    if len(header) < 7 or header[0].upper() != "FID" or header[1].upper() != "IID":
        raise ValueError(f"unexpected header in {path}.")
    body = rows[1:]
    if any(len(r) != len(header) for r in body):
        raise ValueError(f"could not parse the genotype columns of {path}.")
    snp_cols = header[6:]
    snp, allele = [], []
    for name in snp_cols:
        cut = name.rfind("_")
        snp.append(name if cut < 0 else name[:cut])
        allele.append("" if cut < 0 else name[cut + 1:])
    X = np.array([[_float(v) for v in r[6:]] for r in body], dtype=float).T.reshape(len(snp_cols), len(body))
    pheno = np.array([_float(r[5]) for r in body])
    return Raw(X=X, snp=snp, counted_allele=allele, fid=[r[0] for r in body], iid=[r[1] for r in body],
               sex=np.array([_float(r[4]) for r in body]), pheno=pheno, y=_case_control(pheno))


@dataclass
class Assoc:
    snp: List[str]
    chr: List[str]
    a1: List[str]
    bp: np.ndarray
    nmiss: np.ndarray
    odds_ratio: np.ndarray
    beta: np.ndarray                # log odds ratio
    stat: np.ndarray
    p: np.ndarray


def read_plink_assoc(path: PathLike, test: str = "ADD") -> Assoc:
    """PLINK ``--logistic`` report; columns located by name, rows of ``test`` only."""
    rows = _rows(path)
    if not rows:
        raise ValueError(f"empty file {path}.")
    cols = {c.upper(): k for k, c in enumerate(rows[0])}
    if "SNP" not in cols or "P" not in cols:
        raise ValueError(f"columns SNP and P are required in {path}.")
    body = [r for r in rows[1:] if "TEST" not in cols or r[cols["TEST"]].upper() == test.upper()]

    def text(name: str) -> List[str]:
        return [r[cols[name]] for r in body] if name in cols else [""] * len(body)

    def num(name: str) -> np.ndarray:
        return np.array([_float(r[cols[name]]) for r in body]) if name in cols else np.full(len(body), np.nan)

    if "OR" in cols:
        odds = num("OR")
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.log(odds)
    elif "BETA" in cols:
        beta = num("BETA")
        odds = np.exp(beta)
    else:
        odds = beta = np.full(len(body), np.nan)
    return Assoc(snp=text("SNP"), chr=text("CHR"), a1=text("A1"), bp=num("BP"), nmiss=num("NMISS"),
                 odds_ratio=odds, beta=beta, stat=num("STAT"), p=num("P"))


@dataclass
class MatchedAssoc:
    p: np.ndarray
    beta: np.ndarray
    odds_ratio: np.ndarray
    found: np.ndarray
    flipped: np.ndarray


def match_plink_assoc(assoc: Assoc, snp_id: Sequence[str], a1: Optional[Sequence[str]] = None,
                      a2: Optional[Sequence[str]] = None) -> MatchedAssoc:
    """Align a PLINK ``--logistic`` report with the genotype matrix.

    The sign of the log odds ratio is aligned to the counted allele ``a1``:
    ``Assoc.A1 == a1`` keeps it, ``== a2`` negates it, otherwise the SNP is
    excluded (NaN). P-values do not depend on the coded allele.
    """
    loc_of = {}
    for k, x in enumerate(assoc.snp):
        loc_of.setdefault(x, k)
    s = len(snp_id)
    loc = np.array([loc_of.get(x, -1) for x in snp_id], dtype=np.int64)
    found = loc >= 0
    p = np.full(s, np.nan)
    beta0 = np.full(s, np.nan)
    p[found] = assoc.p[loc[found]]
    beta0[found] = assoc.beta[loc[found]]
    flipped = np.zeros(s, dtype=bool)
    if not a1:
        beta = beta0
    else:
        beta = np.full(s, np.nan)
        assoc_a1 = np.array([assoc.a1[k].upper() if k >= 0 else "" for k in loc])
        same = found & (assoc_a1 == np.array([x.upper() for x in a1]))
        swapped = found & (assoc_a1 == np.array([x.upper() for x in a2])) & ~same
        beta[same] = beta0[same]
        beta[swapped] = -beta0[swapped]
        flipped = swapped
        bad = found & ~same & ~swapped
        p[bad] = np.nan
        if bad.any():
            warnings.warn(f"{int(bad.sum())} SNPs have an allele that matches neither A1 nor A2; "
                          "they are excluded.", RuntimeWarning, stacklevel=2)
    return MatchedAssoc(p=p, beta=beta, odds_ratio=np.exp(beta), found=found, flipped=flipped)


@dataclass
class PlinkEpistasis:
    """``--epistasis`` report in matrix form (input of ``phenotypic_network``)."""

    or_int: np.ndarray
    stat: np.ndarray
    p: np.ndarray
    pairs: np.ndarray
    num_skipped: int


def read_plink_epistasis(path: PathLike, snp_ids: Sequence[str]) -> PlinkEpistasis:
    """PLINK 1.9 ``--epistasis`` report (``.epi.cc``): columns SNP1 SNP2 OR_INT STAT P."""
    rows = _rows(path)
    if not rows:
        raise ValueError(f"empty file {path}.")
    cols = {c.upper(): k for k, c in enumerate(rows[0])}
    for need in ("SNP1", "SNP2", "OR_INT", "STAT", "P"):
        if need not in cols:
            raise ValueError(f"column {need} not found in {path} (OR_INT requires PLINK --epistasis).")
    index = {x: j for j, x in enumerate(snp_ids)}
    pairs, values = [], []
    skipped = 0
    for r in rows[1:]:
        j1, j2 = index.get(r[cols["SNP1"]], -1), index.get(r[cols["SNP2"]], -1)
        if j1 < 0 or j2 < 0 or j1 == j2:
            skipped += 1
            continue
        pairs.append((j1, j2))
        values.append((_float(r[cols["OR_INT"]]), _float(r[cols["STAT"]]), _float(r[cols["P"]])))
    s = len(snp_ids)
    pairs_arr = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    vals = np.array(values, dtype=float).reshape(-1, 3)
    return PlinkEpistasis(or_int=pairs_to_matrix(pairs_arr, vals[:, 0], s),
                          stat=pairs_to_matrix(pairs_arr, vals[:, 1], s),
                          p=pairs_to_matrix(pairs_arr, vals[:, 2], s), pairs=pairs_arr, num_skipped=skipped)


def read_snp_gene_relation(path: PathLike, header: Union[str, bool] = "auto") -> Tuple[List[str], List[str]]:
    """SNP-gene relations, ``'SNP GENE [...]'`` per line (e.g. from dbSNP).

    ``header='auto'`` drops the first line when its first field has no digit.
    """
    rows = [r for r in _rows(path) if len(r) >= 2]
    if not rows:
        raise ValueError(f"no relation found in {path}.")
    has_header = (re.search(r"\d", rows[0][0]) is None) if header == "auto" else bool(header)
    if has_header:
        rows = rows[1:]
    return [r[0] for r in rows], [r[1] for r in rows]


def read_edge_list(path: PathLike, node_ids: Optional[Sequence[str]] = None, header: Union[str, bool] = "auto",
                   weight_scale: float = 1.0, min_weight: float = 0.0) -> Tuple[sp.csr_matrix, List[str]]:
    """Undirected weighted network from ``'NodeA NodeB [Weight]'`` lines (e.g. STRING).

    ``header='auto'`` treats the first line as a header when its third field is
    not numeric. Duplicated or reciprocal edges keep their maximum weight;
    self-loops and weights below ``min_weight`` (or zero) are dropped. Without a
    weight field every edge has weight 1.
    """
    rows = _rows(path)
    if not rows:
        raise ValueError(f"empty file {path}.")
    num_field = len(rows[0])
    if num_field < 2:
        raise ValueError(f"each line of {path} needs at least two fields.")
    has_header = (num_field >= 3 and np.isnan(_float(rows[0][2]))) if header == "auto" else bool(header)
    if has_header:
        rows = rows[1:]
    if any(len(r) != num_field for r in rows):
        raise ValueError(f"inconsistent number of fields in {path}.")
    a = [r[0] for r in rows]
    b = [r[1] for r in rows]
    w = (np.array([_float(r[2]) for r in rows]) if num_field >= 3 else np.ones(len(rows))) * weight_scale
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("edge weights must be finite and non-negative.")
    if node_ids is None:
        node_ids = sorted(set(a) | set(b))
    node_ids = list(node_ids)
    w = np.where(w >= min_weight, w, 0.0)
    return edge_matrix(a, b, w, node_ids), node_ids
