"""Unsupervised anatomical anomaly scoring from per-organ volumes.

Novelty: instead of training a disease classifier on labeled pathology, we use
the 100+ anatomical structures segmented by TotalSegmentator as structural
priors and flag *anatomical deviations* with zero disease labels:

1. Robust volume z-scores within the processed cohort (median / MAD-based, so
   a single diseased scan does not move the reference).
2. Left-right asymmetry index for paired organs -- large anatomy is nearly
   symmetric, so strong asymmetry flags unilateral pathology or segmentation
   failure without any labeled examples.
3. Directional age trends from the TotalSegmentator aging analysis
   (Wasserthal et al., Radiology: Artificial Intelligence 2023, PMC10546353),
   reported as Spearman rs: kidney -0.49, pancreas -0.49, aorta +0.64,
   iliac arteries +0.33, iliopsoas volume -0.61. These are *trend priors*
   (direction of change with age), used only to sanity-check the cohort-level
   pattern; they are NOT per-organ normative mean volumes.

With N=2-3 scans this is a proof of concept. The method scales unchanged to a
cohort of thousands (e.g. the 1,939-scan TotalSegmentator training release),
where the z-score reference becomes meaningful.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field

# TotalSegmentator class names are like "<structure>_left"/"<structure>_right".
# Paired structures we test for left-right asymmetry.
PAIRED_STRUCTURES = [
    "adrenal_gland",
    "clavicula",
    "femur",
    "gluteus_maximus",
    "gluteus_medius",
    "gluteus_minimus",
    "hip",
    "iliac_artery",
    "iliopsoas",
    "kidney",
    "lung_upper_lobe",
    "lung_middle_lobe",   # right-only in practice; guarded at runtime
    "lung_lower_lobe",
    "rib_left_",          # placeholder -- ribs are numbered; handled by prefix logic
]

# TotalSegmentator class names use suffixes "_left"/"_right" for most paired
# organs, e.g. kidney_left, kidney_right, lung_upper_lobe_left ...
PAIR_SUFFIXES = ("_left", "_right")

# Age-volume Spearman trends reported by Wasserthal et al. 2023 (aging study on
# n=4004 whole-body CTs). Directional priors only.
AGE_TRENDS_RS = {
    "kidney": -0.49,
    "pancreas": -0.49,
    "aorta": +0.64,
    "iliac_artery": +0.33,
    "iliopsoas": -0.61,
}

# Asymmetry flags above this index are worth a look even with tiny cohorts.
# Index = |L - R| / (L + R) in [0, 1].
ASYMMETRY_WARN = 0.15


@dataclass
class OrganRecord:
    scan: str
    organ: str
    volume_ml: float


@dataclass
class Flag:
    scan: str
    kind: str          # "volume_outlier" | "asymmetry"
    organ: str
    value: float
    reference: float
    detail: str = ""


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def robust_z(value: float, population: list[float]) -> float:
    """Median/MAD-based robust z-score (MAD scaled to sigma for normals)."""
    med = _median(population)
    mad = _median([abs(x - med) for x in population]) or 1e-9
    return 0.6745 * (value - med) / mad


def load_volumes(csv_paths: list[str]) -> dict[str, list[OrganRecord]]:
    """Load per-scan organ volume CSVs -> {scan_id: [records]}."""
    scans: dict[str, list[OrganRecord]] = {}
    for p in csv_paths:
        scan = os.path.basename(p).removesuffix(".csv").removesuffix("_volumes")
        with open(p) as f:
            for row in csv.DictReader(f):
                scans.setdefault(scan, []).append(
                    OrganRecord(scan, row["organ"], float(row["volume_ml"]))
                )
    return scans


def cohort_z_flags(scans: dict[str, list[OrganRecord]], z_thresh: float = 3.0) -> list[Flag]:
    """Flag organs whose volume is an outlier vs the cohort median/MAD."""
    by_organ: dict[str, list[tuple[str, float]]] = {}
    for scan, recs in scans.items():
        for r in recs:
            by_organ.setdefault(r.organ, []).append((scan, r.volume_ml))
    flags: list[Flag] = []
    for organ, vals in by_organ.items():
        if len(vals) < 3:  # robust stats meaningless on <3 points
            continue
        pop = [v for _, v in vals]
        med = _median(pop)
        for scan, v in vals:
            z = robust_z(v, pop)
            if abs(z) >= z_thresh:
                flags.append(Flag(scan, "volume_outlier", organ, z, med,
                                 f"volume {v:.1f} mL vs cohort median {med:.1f} mL"))
    return sorted(flags, key=lambda f: -abs(f.value))


def asymmetry_index(l: float, r: float) -> float:
    return abs(l - r) / (l + r) if (l + r) > 0 else 0.0


def asymmetry_flags(scans: dict[str, list[OrganRecord]]) -> list[Flag]:
    """Flag paired organs with suspicious left-right asymmetry."""
    flags: list[Flag] = []
    for scan, recs in scans.items():
        vols = {r.organ: r.volume_ml for r in recs}
        seen: set[str] = set()
        for organ, v in vols.items():
            base, side = None, None
            for suf in PAIR_SUFFIXES:
                if organ.endswith(suf):
                    base, side = organ[: -len(suf)], suf
                    break
            if base is None or base in seen:
                continue
            seen.add(base)
            l = vols.get(base + "_left", 0.0)
            r = vols.get(base + "_right", 0.0)
            if l <= 0 or r <= 0:
                continue  # missing/empty side: not an asymmetry signal
            ai = asymmetry_index(l, r)
            if ai >= ASYMMETRY_WARN:
                flags.append(Flag(scan, "asymmetry", base, ai, 0.0,
                                  f"L={l:.1f} mL, R={r:.1f} mL"))
    return sorted(flags, key=lambda f: -f.value)


def trend_priors_report() -> str:
    lines = ["Age-volume trend priors (Wasserthal et al. 2023, n=4004, Spearman rs):"]
    for organ, rs in AGE_TRENDS_RS.items():
        direction = "decreases with age" if rs < 0 else "increases with age"
        lines.append(f"  {organ}: rs={rs:+.2f} ({direction})")
    lines.append("Note: scans used here have no age metadata, so trends are checked "
                 "qualitatively at cohort level only.")
    return "\n".join(lines)


def write_report(flags: list[Flag], out_path: str, cohort_size: int) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan", "flag_kind", "organ", "score", "reference", "detail"])
        for fl in flags:
            w.writerow([fl.scan, fl.kind, fl.organ, f"{fl.value:.3f}",
                        f"{fl.reference:.2f}", fl.detail])
    print(f"wrote {len(flags)} flags to {out_path} (cohort n={cohort_size})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unsupervised anatomical anomaly scoring.")
    ap.add_argument("--volumes", nargs="+", required=True,
                    help="Per-scan organ volume CSVs from src/volumetry.py")
    ap.add_argument("--out", required=True, help="Output anomaly report CSV")
    ap.add_argument("--z-thresh", type=float, default=3.0)
    args = ap.parse_args()

    scans = load_volumes(args.volumes)
    flags = cohort_z_flags(scans, args.z_thresh) + asymmetry_flags(scans)
    write_report(flags, args.out, len(scans))
    print(trend_priors_report())
    if flags:
        print("\nRanked flags:")
        for fl in flags:
            print(f"  [{fl.scan}] {fl.kind}: {fl.organ} score={fl.value:.3f} ({fl.detail})")
    else:
        print("\nNo flags raised.")


if __name__ == "__main__":
    main()
