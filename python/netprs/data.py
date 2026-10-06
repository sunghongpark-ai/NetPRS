"""NetPRS input data: the container and the single-file CSV format.

CSV format (``dataset/sample.csv``)
-----------------------------------
UTF-8 text with the header ``table,id,key,value`` and one record per line. The
four tables hold every input of NetPRS (Table I of the paper: participant
genotype, SNP-gene relation and gene-gene interaction):

==========  ============  ===========  ==========================================
table       id            key          value
==========  ============  ===========  ==========================================
subject     subject ID    diagnosis    1 = AD, 0 = non-AD (required)
subject     subject ID    validation   1 = validation cohort, 0 = discovery
                                       (optional, default 0)
genotype    subject ID    SNP ID       dosage of the counted allele (0, 1, 2);
                                       empty or ``NA`` = missing call
snp_gene    SNP ID        gene ID      relation weight (> 0; 1 for dbSNP links)
gene_gene   gene ID       gene ID      interaction weight (>= 0, e.g. STRING
                                       combined_score / 1000)
==========  ============  ===========  ==========================================

Rules: subjects keep the order of their first ``subject`` row and SNPs the
order of their first ``genotype`` row; genes are sorted. A (subject, SNP) pair
without a genotype row is a missing call. ``snp_gene``/``gene_gene`` rows are
optional (without them the genomic network W_G cannot be built). Duplicated or
reciprocal ``gene_gene`` rows keep their maximum weight; self-loops and zero
weights add no edge. Blank lines, a byte-order mark, CRLF line ends, white
space around fields and fields enclosed in double quotes are accepted; a field
must not contain a comma or a double quote. Errors name the offending line.

The MATLAB functions ``ReadNetPRSCSV.m``/``WriteNetPRSCSV.m`` implement the same
rules, and both writers produce byte-identical files.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

__all__ = ["NetPRSData", "read_netprs_csv", "write_netprs_csv", "edge_matrix"]

PathLike = Union[str, Path]
_HEADER = ["table", "id", "key", "value"]
_TABLES = ("subject", "genotype", "snp_gene", "gene_gene")
_MISSING = ("", "NA", "NAN")
_BAD_ID = re.compile(r'[,"\n\r]|^\s|\s$')


@dataclass
class NetPRSData:
    """Inputs of the NetPRS pipeline (the ``Data`` struct of the MATLAB version).

    Attributes
    ----------
    genotype : ndarray, shape (s, n)
        Allele dosage (SNPs x subjects, as ``X`` in the paper); NaN = missing.
    diagnosis : ndarray, shape (n,)
        1 = AD, 0 = non-AD.
    is_validation : ndarray of bool, shape (n,)
        Validation-cohort membership (the other subjects form the discovery cohort).
    subject_id, snp_id, gene_id : list of str
    ggi : scipy.sparse matrix, shape (m, m)
        Symmetric gene-gene interaction network with zero diagonal.
    rel_snp, rel_gene : list of str
        SNP-gene relations, with weights ``rel_weight``.
    allele1, allele2 : list of str, optional
        Counted and other allele of every SNP (PLINK input only).
    source : str
        ``"csv"``, ``"synthetic"`` or ``"plink"``.
    """

    genotype: np.ndarray
    diagnosis: np.ndarray
    is_validation: np.ndarray
    subject_id: List[str]
    snp_id: List[str]
    gene_id: List[str] = field(default_factory=list)
    ggi: sp.spmatrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    rel_snp: List[str] = field(default_factory=list)
    rel_gene: List[str] = field(default_factory=list)
    rel_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    allele1: Optional[List[str]] = None
    allele2: Optional[List[str]] = None
    source: str = ""

    def __post_init__(self) -> None:
        self.genotype = np.asarray(self.genotype, dtype=float)
        self.diagnosis = np.asarray(self.diagnosis, dtype=float).ravel()
        self.is_validation = np.asarray(self.is_validation, dtype=bool).ravel()
        self.rel_weight = np.asarray(self.rel_weight, dtype=float).ravel()
        self.ggi = sp.csr_matrix(self.ggi)
        s, n = self.genotype.shape if self.genotype.ndim == 2 else (-1, -1)
        if s < 0 or len(self.snp_id) != s or len(self.subject_id) != n:
            raise ValueError("genotype must be len(snp_id) x len(subject_id).")
        if self.diagnosis.size != n or self.is_validation.size != n:
            raise ValueError("diagnosis and is_validation need one value per subject.")
        if len(self.rel_gene) != len(self.rel_snp) or self.rel_weight.size != len(self.rel_snp):
            raise ValueError("rel_snp, rel_gene and rel_weight must have equal length.")
        if self.ggi.shape != (len(self.gene_id), len(self.gene_id)):
            raise ValueError("ggi must be len(gene_id) x len(gene_id).")

    @property
    def num_snp(self) -> int:
        return int(self.genotype.shape[0])

    @property
    def num_subject(self) -> int:
        return int(self.genotype.shape[1])

    @property
    def has_genomic_knowledge(self) -> bool:
        return len(self.rel_snp) > 0 and self.ggi.nnz > 0


def edge_matrix(a: Sequence[str], b: Sequence[str], w: np.ndarray,
                node_ids: Sequence[str]) -> sp.csr_matrix:
    """Symmetric sparse adjacency from an edge list.

    Edges with unknown nodes, self-loops or non-positive weights are dropped;
    duplicated or reciprocal edges keep their maximum weight (as ``ReadEdgeList``).
    """
    index = {g: i for i, g in enumerate(node_ids)}
    m = len(node_ids)
    ia = np.array([index.get(x, -1) for x in a], dtype=np.int64)
    ib = np.array([index.get(x, -1) for x in b], dtype=np.int64)
    w = np.asarray(w, dtype=float)
    ok = (ia >= 0) & (ib >= 0) & (ia != ib) & (w > 0)
    lo = np.minimum(ia[ok], ib[ok])
    hi = np.maximum(ia[ok], ib[ok])
    if lo.size == 0:
        return sp.csr_matrix((m, m))
    key = lo * m + hi
    order = np.lexsort((-w[ok], key))           # within a key, the largest weight first
    key_sorted = key[order]
    first = np.ones(key_sorted.size, dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    keep = order[first]
    upper = sp.csr_matrix((w[ok][keep], (lo[keep], hi[keep])), shape=(m, m))
    return (upper + upper.T).tocsr()


def read_netprs_csv(path: PathLike) -> NetPRSData:
    """Read and validate a NetPRS CSV file (see the module docstring)."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        records: List[Tuple[int, List[str]]] = []
        for row in reader:
            if row and any(x.strip() for x in row):
                records.append((reader.line_num, row))
    if not records:
        raise ValueError(f"{path} is empty.")
    header_line, header = records[0]
    if [h.strip().lower() for h in header] != _HEADER:
        raise ValueError(f"{path}: the header must be 'table,id,key,value'.")
    if len(records) == 1:
        raise ValueError(f"{path}: no subject rows.")

    def err(line: int, message: str) -> ValueError:
        return ValueError(f"{path}, line {line}: {message}")

    subj_seen: Dict[str, None] = {}                  # insertion-ordered set
    diagnosis: Dict[str, float] = {}
    validation: Dict[str, float] = {}
    geno_rows: List[Tuple[str, str, str, int]] = []
    rel_rows: List[Tuple[str, str, float]] = []
    ggi_rows: List[Tuple[str, str, float]] = []
    for line, row in records[1:]:
        if len(row) != 4:
            raise err(line, f"expected 4 fields, found {len(row)}.")
        table, rid, key, value = (x.strip() for x in row)
        if any(("," in x) or ('"' in x) for x in (table, rid, key, value)):
            raise err(line, "commas or double quotes inside a field are not supported.")
        table = table.lower()
        if table not in _TABLES:
            raise err(line, f"unknown table '{table}'.")
        if not rid or not key:
            raise err(line, "empty id or key.")
        if table == "subject":
            k = key.lower()
            if k not in ("diagnosis", "validation"):
                raise err(line, f"unknown subject key '{key}'.")
            v = _to_float(value)
            if v not in (0.0, 1.0):
                raise err(line, f"subject {k} must be 0 or 1 (found '{value}').")
            target = diagnosis if k == "diagnosis" else validation
            if rid in target:
                raise err(line, f"duplicated {k} of subject '{rid}'.")
            target[rid] = v
            subj_seen.setdefault(rid, None)
        elif table == "genotype":
            geno_rows.append((rid, key, value, line))
        elif table == "snp_gene":
            w = _to_float(value)
            if not (np.isfinite(w) and w > 0):
                raise err(line, f"snp_gene weight must be positive (found '{value}').")
            rel_rows.append((rid, key, w))
        else:
            w = _to_float(value)
            if not (np.isfinite(w) and w >= 0):
                raise err(line, f"gene_gene weight must be >= 0 (found '{value}').")
            ggi_rows.append((rid, key, w))

    subj_order = list(subj_seen)
    if not subj_order:
        raise ValueError(f"{path}: no subject rows.")
    missing_dx = [x for x in subj_order if x not in diagnosis]
    if missing_dx:
        raise ValueError(f"{path}: subject '{missing_dx[0]}' has no diagnosis.")
    if not geno_rows:
        raise ValueError(f"{path}: no genotype rows.")

    subj_index = {x: i for i, x in enumerate(subj_order)}
    snp_index: Dict[str, int] = {}
    for rid, key, _, line in geno_rows:
        if rid not in subj_index:
            raise err(line, f"genotype of unknown subject '{rid}'.")
        snp_index.setdefault(key, len(snp_index))
    s, n = len(snp_index), len(subj_order)
    genotype = np.full((s, n), np.nan)
    seen = np.zeros((s, n), dtype=bool)
    for rid, key, value, line in geno_rows:
        j, i = snp_index[key], subj_index[rid]
        if seen[j, i]:
            raise err(line, f"duplicated genotype ({rid}, {key}).")
        seen[j, i] = True
        if value.upper() in _MISSING:
            continue
        v = _to_float(value)
        if v not in (0.0, 1.0, 2.0):
            raise err(line, f"genotype must be 0, 1, 2 or empty (found '{value}').")
        genotype[j, i] = v

    genes = sorted({g for _, g, _ in rel_rows} | {a for a, _, _ in ggi_rows} | {b for _, b, _ in ggi_rows})
    ggi = edge_matrix([a for a, _, _ in ggi_rows], [b for _, b, _ in ggi_rows],
                      np.array([w for _, _, w in ggi_rows], dtype=float), genes)
    return NetPRSData(
        genotype=genotype,
        diagnosis=np.array([diagnosis[x] for x in subj_order], dtype=float),
        is_validation=np.array([validation.get(x, 0.0) == 1.0 for x in subj_order], dtype=bool),
        subject_id=subj_order, snp_id=list(snp_index), gene_id=genes, ggi=ggi,
        rel_snp=[a for a, _, _ in rel_rows], rel_gene=[b for _, b, _ in rel_rows],
        rel_weight=np.array([w for _, _, w in rel_rows], dtype=float), source="csv")


def write_netprs_csv(path: PathLike, data: NetPRSData) -> None:
    """Write ``data`` in the NetPRS CSV format (byte-identical to ``WriteNetPRSCSV.m``).

    Rows: subject (diagnosis, validation), genotype (subject-major; missing calls
    as empty values), snp_gene, gene_gene (each edge of the symmetric ``ggi``
    once, upper triangle in row-major order). Numbers are printed with ``%.10g``;
    lines end with LF. Everything the reader would reject (dosages other than
    0/1/2, duplicated or malformed identifiers, an asymmetric ``ggi``) is rejected.
    """
    if np.any(~np.isin(data.diagnosis, (0.0, 1.0))):
        raise ValueError("diagnosis must be 0 or 1.")
    geno = data.genotype
    observed = geno[~np.isnan(geno)]
    if np.any(~np.isin(observed, (0.0, 1.0, 2.0))):
        raise ValueError("genotype must contain the dosages 0, 1, 2 or NaN (missing).")
    rel_weight = data.rel_weight
    if np.any(~np.isfinite(rel_weight) | (rel_weight <= 0)):
        raise ValueError("relation weights must be finite and positive.")
    ggi = sp.csr_matrix(data.ggi)
    ggi.eliminate_zeros()
    if np.any(~np.isfinite(ggi.data) | (ggi.data < 0)):
        raise ValueError("ggi must be finite and non-negative.")
    asym = abs(ggi - ggi.T)
    if asym.nnz and asym.max() > 1e-12 * abs(ggi).max():
        raise ValueError("ggi must be symmetric (each edge is written once).")
    upper = sp.triu(ggi, k=1).tocoo()
    _check_ids(list(data.subject_id) + list(data.snp_id) + list(data.rel_snp)
               + list(data.rel_gene) + (list(data.gene_id) if upper.nnz else []))
    for name, ids in (("subject", data.subject_id), ("SNP", data.snp_id),
                      ("gene", data.gene_id if upper.nnz else [])):
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicated {name} identifiers.")

    out: List[str] = ["table,id,key,value"]
    for i, sid in enumerate(data.subject_id):
        out.append(f"subject,{sid},diagnosis,{_fmt(data.diagnosis[i])}")
        out.append(f"subject,{sid},validation,{_fmt(1.0 if data.is_validation[i] else 0.0)}")
    texts = {}                                        # dosage -> text (few distinct values)
    for i, sid in enumerate(data.subject_id):
        prefix = f"genotype,{sid},"
        for j, snp in enumerate(data.snp_id):
            v = geno[j, i]
            if np.isnan(v):
                out.append(f"{prefix}{snp},")
            else:
                text = texts.get(v)
                if text is None:
                    text = texts[v] = _fmt(v)
                out.append(f"{prefix}{snp},{text}")
    for snp, gene, w in zip(data.rel_snp, data.rel_gene, rel_weight):
        out.append(f"snp_gene,{snp},{gene},{_fmt(w)}")
    order = np.lexsort((upper.col, upper.row))
    for r, c, w in zip(upper.row[order], upper.col[order], upper.data[order]):
        out.append(f"gene_gene,{data.gene_id[r]},{data.gene_id[c]},{_fmt(w)}")
    with Path(path).open("w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")


def _fmt(x: float) -> str:
    return f"{float(x):.10g}"


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _check_ids(ids: Sequence[str]) -> None:
    for x in ids:
        if not x or _BAD_ID.search(x):
            raise ValueError(f"Identifier '{x}' is empty, contains a comma, a double quote or a "
                             "line break, or starts/ends with white space.")
