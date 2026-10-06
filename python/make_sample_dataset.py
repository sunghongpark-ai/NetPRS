"""Regenerate ``dataset/sample.csv`` (synthetic NetPRS data) or check it.

    python make_sample_dataset.py                 # writes ../dataset/sample.csv
    python make_sample_dataset.py --check         # compares with the existing file
    python make_sample_dataset.py --output x.csv --num-subject 2000 --seed 7

The default options reproduce the published file byte for byte; the MATLAB
statements ``[Data, Truth] = GenerateSyntheticData(); WriteNetPRSCSV(File, Data)``
write the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from netprs.data import write_netprs_csv
from netprs.synthetic import generate_synthetic_data

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "dataset" / "sample.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--check", action="store_true", help="compare a fresh copy with --output instead of writing")
    p.add_argument("--num-subject", type=int, default=1000)
    p.add_argument("--num-validation", type=int, default=300)
    p.add_argument("--num-snp", type=int, default=300)
    p.add_argument("--num-gene", type=int, default=120)
    p.add_argument("--num-module", type=int, default=6)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args(argv)
    data, _ = generate_synthetic_data(num_subject=args.num_subject, num_validation=args.num_validation,
                                      num_snp=args.num_snp, num_gene=args.num_gene, num_module=args.num_module,
                                      seed=args.seed)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "sample.csv"
            write_netprs_csv(fresh, data)
            same = args.output.exists() and fresh.read_bytes() == args.output.read_bytes()
            print(f"fresh    sha256 {sha256(fresh)}")
            if args.output.exists():
                print(f"existing sha256 {sha256(args.output)}  ({args.output})")
            print("identical" if same else "DIFFERENT")
            return 0 if same else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_netprs_csv(args.output, data)
    print(f"wrote {args.output} ({data.num_subject} subjects, {data.num_snp} SNPs, {len(data.gene_id)} genes); "
          f"sha256 {sha256(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
