"""Per-organ volumetry from TotalSegmentator masks.

Each TotalSegmentator class is stored as an individual binary NIfTI mask
``<class_name>.nii.gz`` inside the segmentation output directory. This module
aggregates voxel counts and converts them to millilitres using the image
spacing from the NIfTI header.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import nibabel as nib

# Classes that are not volumetric anatomy for our purpose (skip in reports)
NON_ANATOMY = {
    "body", "body_trunc", "body_extremities", "heart",  # aggregates / regions
}


@dataclass
class OrganVolume:
    name: str
    voxels: int
    volume_ml: float


def list_masks(seg_dir: str) -> list[str]:
    """Return absolute paths of all .nii.gz masks in a TotalSegmentator output dir."""
    return sorted(
        os.path.join(seg_dir, f)
        for f in os.listdir(seg_dir)
        if f.endswith(".nii.gz")
    )


def mask_volume(path: str) -> OrganVolume:
    """Compute voxel count and volume (mL) for one binary mask file."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    vox = img.header.get_zooms()[:3]
    voxel_ml = float(np.prod(vox)) / 1000.0  # mm^3 -> mL
    name = os.path.basename(path).removesuffix(".nii.gz")
    nonzero = int(np.count_nonzero(data))
    return OrganVolume(name=name, voxels=nonzero, volume_ml=nonzero * voxel_ml)


def scan_volumes(seg_dir: str, exclude: set[str] | None = None) -> list[OrganVolume]:
    """Compute volumes for every mask in ``seg_dir``."""
    exclude = (exclude or set()) | NON_ANATOMY
    vols = [mask_volume(p) for p in list_masks(seg_dir)]
    return [v for v in vols if v.name not in exclude and v.voxels > 0]


def write_csv(volumes: list[OrganVolume], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organ", "voxels", "volume_ml"])
        for v in sorted(volumes, key=lambda x: -x.volume_ml):
            w.writerow([v.name, v.voxels, f"{v.volume_ml:.2f}"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute per-organ volumes from TotalSegmentator masks.")
    ap.add_argument("--seg-dir", required=True, help="Directory with <class>.nii.gz masks")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    vols = scan_volumes(args.seg_dir)
    write_csv(vols, args.out)
    print(f"wrote {len(vols)} organ volumes to {args.out}")
