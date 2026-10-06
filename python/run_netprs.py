"""NetPRS: SNP interaction aware network-based polygenic risk score (command line).

Python counterpart of ``matlab/NetPRS.m``: runs Stage 1 (SNP screening, W_P,
W_G) and Stage 2 (NetPRS with repeated cross-validation, ablation, wPRS
baseline, interpretation) and writes the result tables.

    python run_netprs.py                                  # ../dataset/sample.csv
    python run_netprs.py --data my_cohort.csv --output my_result
    python run_netprs.py --num-iter 2 --no-ablation       # quick run

Park S, Lee D, Kim J, et al. "NetPRS: SNP interaction aware network-based
polygenic risk score for Alzheimer's disease." IEEE EMBS BHI 2024.
doi:10.1109/BHI62660.2024.10913658
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from netprs import export_results, netprs_defaults, run_netprs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    config, params = netprs_defaults()
    p = argparse.ArgumentParser(description="Run the complete NetPRS analysis.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=config.csv_file, help="input file in the NetPRS CSV format")
    p.add_argument("--output", default=config.output_dir, help="folder of the result tables")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(config.level_thresholds),
                   help="-log10(P) cut-offs of the SNP levels")
    p.add_argument("--levels", type=int, nargs="+", default=list(config.levels), help="levels to analyse")
    p.add_argument("--effects", default=config.effects, choices=["I", "P", "G", "IP", "IG", "PG", "IPG"],
                   help="effects of the NetPRS model")
    p.add_argument("--num-iter", type=int, default=params.num_iter, help="repetitions of stratified K-fold CV")
    p.add_argument("--num-fold", type=int, default=params.num_fold, help="number of folds K")
    p.add_argument("--max-epoch", type=int, default=params.max_epoch, help="ADAM steps")
    p.add_argument("--learn-rate", type=float, default=params.learn_rate)
    p.add_argument("--reg-coeff", type=float, default=params.reg_coeff, help="L2 coefficient delta")
    p.add_argument("--seed", type=int, default=params.seed, help="seed of the folds and of beta")
    p.add_argument("--no-qc", action="store_true", help="skip quality control (data already QC-ed)")
    p.add_argument("--no-ablation", action="store_true", help="NetPRS only (7x faster)")
    p.add_argument("--no-baseline", action="store_true", help="skip the wPRS baseline")
    p.add_argument("--no-interpretation", action="store_true", help="skip the final model and SHAP")
    p.add_argument("--workers", type=int, default=0, help="parallel processes for the CV fits")
    p.add_argument("--quiet", action="store_true", help="no progress messages")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config, params = netprs_defaults()
    config = replace(config, data_source="csv", csv_file=str(Path(args.data)), output_dir=str(Path(args.output)),
                     level_thresholds=tuple(args.thresholds), levels=tuple(args.levels), effects=args.effects,
                     run_qc=not args.no_qc, run_ablation=not args.no_ablation, run_baseline=not args.no_baseline,
                     run_interpretation=not args.no_interpretation, num_workers=args.workers,
                     verbose=not args.quiet)
    params = replace(params, num_iter=args.num_iter, num_fold=args.num_fold, max_epoch=args.max_epoch,
                     learn_rate=args.learn_rate, reg_coeff=args.reg_coeff, seed=args.seed)
    results, info = run_netprs(config, params)
    files = export_results(results, info)
    if config.verbose:
        print(f"\nResults written to {config.output_dir}:")
        for f in files:
            print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
