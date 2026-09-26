"""Cohort-level organ volume summary: robust per-organ statistics across scans.

Reads per-scan volume CSVs (from src/volumetry.py) and computes, for each
organ, the cohort size n, median volume, and interquartile range. This is the
summary a reader needs to judge the anomaly flags -- and the reference
distribution a future cohort z-score pass would be scored against.

Usage:
    python src/cohort_summary.py results/case_00000_volumes.csv \
        results/case_00001_volumes.csv --out results/cohort_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

from anomaly_scoring import load_volumes


@dataclass
class CohortStat:
    organ: str
    n: int
    median_ml: float
    q1_ml: float
    q3_ml: float

    @property
    def iqr_ml(self) -> float:
        return self.q3_ml - self.q1_ml


def _quantile(xs: list[float], q: float) -> float:
    """Linear-interpolation quantile (same convention as numpy default)."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[min(lo + 1, len(s) - 1)] * frac


def cohort_stats(csv_paths: list[str]) -> list[CohortStat]:
    """Median/Q1/Q3 volume per organ across all scans that contain it."""
    scans = load_volumes(csv_paths)
    by_organ: dict[str, list[float]] = {}
    for recs in scans.values():
        for r in recs:
            by_organ.setdefault(r.organ, []).append(r.volume_ml)
    out = []
    for organ, vals in by_organ.items():
        out.append(CohortStat(
            organ=organ,
            n=len(vals),
            median_ml=_quantile(vals, 0.5),
            q1_ml=_quantile(vals, 0.25),
            q3_ml=_quantile(vals, 0.75),
        ))
    return sorted(out, key=lambda s: -s.median_ml)


def write_summary_csv(stats: list[CohortStat], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organ", "n_scans", "median_ml", "q1_ml", "q3_ml", "iqr_ml"])
        for s in stats:
            w.writerow([s.organ, s.n, f"{s.median_ml:.2f}", f"{s.q1_ml:.2f}",
                        f"{s.q3_ml:.2f}", f"{s.iqr_ml:.2f}"])
    print(f"wrote {len(stats)} organ summaries to {out_path}")


def markdown_table(stats: list[CohortStat], organs: list[str]) -> str:
    """Render a markdown table for selected organs (README-ready)."""
    by_name = {s.organ: s for s in stats}
    lines = ["| organ | n | median mL | IQR mL |",
             "|---|---|---|---|"]
    for organ in organs:
        s = by_name.get(organ)
        if s is None:
            lines.append(f"| {organ} | 0 | — | — |")
        else:
            lines.append(f"| {organ} | {s.n} | {s.median_ml:.1f} | "
                         f"{s.q1_ml:.1f}–{s.q3_ml:.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cohort organ-volume summary (median/IQR per organ).")
    ap.add_argument("volumes_csv", nargs="+",
                    help="Per-scan volume CSVs from src/volumetry.py")
    ap.add_argument("--out", default="results/cohort_summary.csv",
                    help="Output summary CSV path")
    ap.add_argument("--organs", nargs="*", default=[],
                    help="Organs to print as a markdown table")
    args = ap.parse_args()

    stats = cohort_stats(args.volumes_csv)
    write_summary_csv(stats, args.out)
    print(f"\ncohort: {len(load_volumes(args.volumes_csv))} scans, "
          f"{len(stats)} organs")
    if args.organs:
        print()
        print(markdown_table(stats, args.organs))
    else:
        print("\nLargest organs by cohort median:")
        for s in stats[:10]:
            print(f"  {s.organ}: n={s.n} median={s.median_ml:.1f} mL "
                  f"IQR={s.q1_ml:.1f}-{s.q3_ml:.1f}")


if __name__ == "__main__":
    main()
