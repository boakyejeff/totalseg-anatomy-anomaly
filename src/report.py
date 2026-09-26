"""End-to-end report script: volumetry + anomaly scoring for N scans.

Usage (after running TotalSegmentator, see README):

    python src/report.py --scan segs/case_00000=case_00000 \
                         --scan segs/case_00001=case_00001 \
                         --out results/anomaly_report.csv

    # or from precomputed per-scan volume CSVs (skips volumetry):
    python src/report.py --volumes-csv results/case_00000_volumes.csv \
                         --volumes-csv results/case_00001_volumes.csv \
                         --out results/anomaly_report.csv

Cohort-level statistics (volume z-scores) use ALL scans given; robust
z-scores require n>=3 (see anomaly_scoring.cohort_z_flags).
"""
from __future__ import annotations

import argparse
import os

from volumetry import scan_volumes, write_csv
from anomaly_scoring import (
    load_volumes, cohort_z_flags, asymmetry_flags, write_report,
    trend_priors_report, flag_recurrence,
    all_asymmetry_indices, write_asymmetry_csv,
)


def volumes_for_scan(seg_dir: str, scan_id: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vols = scan_volumes(seg_dir)
    path = os.path.join(out_dir, f"{scan_id}_volumes.csv")
    write_csv(vols, path)
    print(f"{scan_id}: {len(vols)} organ volumes")
    return path


def parse_scan_pair(pair: str) -> tuple[str, str]:
    """Parse 'seg_dir=scan_id' into (seg_dir, scan_id)."""
    if "=" not in pair:
        raise argparse.ArgumentTypeError(
            f"expected DIR=SCAN_ID, got {pair!r}")
    d, _, sid = pair.partition("=")
    if not d or not sid:
        raise argparse.ArgumentTypeError(
            f"expected DIR=SCAN_ID, got {pair!r}")
    return d, sid


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Anatomical anomaly report from TotalSegmentator masks.")
    ap.add_argument("--scan", action="append", metavar="DIR=SCAN_ID",
                    help="TotalSegmentator output dir + scan id; repeatable")
    ap.add_argument("--volumes-csv", action="append", metavar="PATH",
                    help="Precomputed per-scan volume CSV; repeatable "
                         "(alternative to --scan)")
    ap.add_argument("--out-dir", default="results",
                    help="Where per-scan volume CSVs go")
    ap.add_argument("--out", default="results/anomaly_report.csv",
                    help="Report CSV path")
    ap.add_argument("--z-thresh", type=float, default=3.0)
    ap.add_argument("--asymmetry-out", default="results/asymmetry_indices.csv",
                    help="Full per-organ asymmetry index CSV path")
    args = ap.parse_args()

    csvs: list[str] = []
    if args.scan:
        for pair in args.scan:
            seg_dir, scan_id = parse_scan_pair(pair)
            csvs.append(volumes_for_scan(seg_dir, scan_id, args.out_dir))
    if args.volumes_csv:
        csvs.extend(args.volumes_csv)
    if not csvs:
        ap.error("give at least one --scan DIR=SCAN_ID or --volumes-csv PATH")

    scans = load_volumes(csvs)
    flags = cohort_z_flags(scans, args.z_thresh) + asymmetry_flags(scans)
    write_report(flags, args.out, len(scans))
    write_asymmetry_csv(all_asymmetry_indices(scans), args.asymmetry_out)
    print(trend_priors_report())

    print("\nTop flags:")
    for fl in flags[:20]:
        print(f"  [{fl.scan}] {fl.kind}: {fl.organ} score={fl.value:.3f} ({fl.detail})")

    rec = flag_recurrence(flags)
    if rec:
        print("\nFlag recurrence across cohort:")
        for kind, organ, n_scans, scan_ids in rec:
            print(f"  {kind} {organ}: {n_scans}/{len(scans)} scans "
                  f"({', '.join(scan_ids)})")


if __name__ == "__main__":
    main()
