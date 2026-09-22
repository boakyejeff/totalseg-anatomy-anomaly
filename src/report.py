"""End-to-end report script: volumetry + anomaly scoring for one or more scans.

Usage (after running TotalSegmentator, see README):

    python src/report.py --seg-dir segs_scan1 --scan-id scan1 --out results/report_scan1.csv
    python src/report.py --seg-dir segs_scan1 --scan-id scan1 \
                         --seg-dir2 segs_scan2 --scan-id2 scan2 --out results/report.csv

Cohort-level statistics (volume z-scores) use ALL volume CSVs passed via
--volumes-csv, or default to the ones produced in this run.
"""
from __future__ import annotations

import argparse
import csv
import os

from volumetry import scan_volumes, write_csv
from anomaly_scoring import (
    load_volumes, cohort_z_flags, asymmetry_flags, write_report,
    trend_priors_report,
)


def volumes_for_scan(seg_dir: str, scan_id: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    vols = scan_volumes(seg_dir)
    path = os.path.join(out_dir, f"{scan_id}_volumes.csv")
    write_csv(vols, path)
    print(f"{scan_id}: {len(vols)} organ volumes")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Anatomical anomaly report from TotalSegmentator masks.")
    ap.add_argument("--seg-dir", required=True, help="Mask dir for scan 1")
    ap.add_argument("--scan-id", required=True, help="ID for scan 1")
    ap.add_argument("--seg-dir2", help="Mask dir for scan 2 (optional)")
    ap.add_argument("--scan-id2", help="ID for scan 2")
    ap.add_argument("--out-dir", default="results", help="Where CSVs go")
    ap.add_argument("--out", default="results/anomaly_report.csv", help="Report CSV path")
    ap.add_argument("--z-thresh", type=float, default=3.0)
    args = ap.parse_args()

    csvs = [volumes_for_scan(args.seg_dir, args.scan_id, args.out_dir)]
    if args.seg_dir2:
        csvs.append(volumes_for_scan(args.seg_dir2, args.scan_id2 or "scan2", args.out_dir))

    scans = load_volumes(csvs)
    flags = cohort_z_flags(scans, args.z_thresh) + asymmetry_flags(scans)
    write_report(flags, args.out, len(scans))
    print(trend_priors_report())
    print("\nTop flags:")
    for fl in flags[:20]:
        print(f"  [{fl.scan}] {fl.kind}: {fl.organ} score={fl.value:.3f} ({fl.detail})")


if __name__ == "__main__":
    main()
