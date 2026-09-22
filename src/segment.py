"""Wrapper for running TotalSegmentator inference on a CT volume.

The TotalSegmentator call itself is a documented manual step (it downloads
~1 GB of nnU-Net weights on first run and takes 15-45 min on CPU):

    TotalSegmentator -i scan.nii.gz -o segs --fast

This module validates inputs/outputs around that step and can also invoke it
as a subprocess.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import nibabel as nib


def check_ct(path: str) -> dict:
    """Basic sanity check: loads NIfTI, reports shape, spacing, dtype."""
    img = nib.load(path)
    zooms = img.header.get_zooms()
    info = {
        "path": path,
        "shape": tuple(img.shape),
        "spacing_mm": tuple(round(float(z), 3) for z in zooms[:3]),
        "dtype": str(img.get_data_dtype()),
        "n_voxels": int(__import__("numpy").prod(img.shape)),
    }
    print(f"CT: {info['shape']} voxels, spacing {info['spacing_mm']} mm, dtype {info['dtype']}")
    return info


def run_totalsegmentator(ct_path: str, out_dir: str, fast: bool = True) -> None:
    """Run TotalSegmentator on ``ct_path`` writing masks to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", ct_path, "-o", out_dir]
    if fast:
        cmd.append("--fast")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def count_masks(out_dir: str) -> int:
    n = len([f for f in os.listdir(out_dir) if f.endswith(".nii.gz")])
    print(f"{n} masks in {out_dir}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="TotalSegmentator segmentation runner.")
    ap.add_argument("--ct", required=True, help="Input CT as .nii.gz")
    ap.add_argument("--out", required=True, help="Output directory for masks")
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--check-only", action="store_true",
                    help="Only validate the input CT, do not run inference")
    args = ap.parse_args()

    check_ct(args.ct)
    if not args.check_only:
        run_totalsegmentator(args.ct, args.out, args.fast)
        count_masks(args.out)


if __name__ == "__main__":
    main()
