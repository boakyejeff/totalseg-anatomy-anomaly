"""Lossless float64 -> int16 conversion for KiTS19 CT scans.

The KiTS19 NIfTIs are stored as float64 although the values are integers in
HU range. This script asserts the value range fits int16, casts, saves as
``<name>_i16.nii.gz`` with a proper int16 NIfTI header, verifies the written
file round-trips exactly, and only then deletes the float64 original.

Usage:
    python src/convert_dtype.py data/           # converts all case_*.nii.gz
    python src/convert_dtype.py data/case_00002.nii.gz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np


def convert(path: str) -> str:
    img = nib.load(path)
    d64 = np.asarray(img.dataobj)
    mn, mx = float(d64.min()), float(d64.max())
    print(f"{os.path.basename(path)}: dtype={d64.dtype}, shape={d64.shape}, "
          f"range=[{mn:.1f}, {mx:.1f}]", flush=True)
    assert mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max, (
        f"range [{mn}, {mx}] does NOT fit int16 -- refusing to cast")
    d16 = d64.astype(np.int16)
    del d64
    assert np.array_equal(np.asarray(nib.load(path).dataobj),
                          d16.astype(np.float64)), "roundtrip mismatch"
    out = path.removesuffix(".nii.gz") + "_i16.nii.gz"
    hdr = img.header.copy()
    hdr.set_data_dtype(np.int16)  # reusing the float64 header would mislabel dtype
    nib.save(nib.Nifti1Image(d16, img.affine, hdr), out)
    back = np.asarray(nib.load(out).dataobj)
    assert back.dtype == np.int16, "header dtype not int16"
    assert np.array_equal(back, d16), "post-write data mismatch"
    del d16
    os.remove(path)
    print(f"  -> {os.path.basename(out)} ({os.path.getsize(out)} bytes); "
          f"original deleted", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lossless float64->int16 NIfTI conversion.")
    ap.add_argument("target", help="NIfTI file or directory containing case_*.nii.gz")
    args = ap.parse_args()

    if os.path.isdir(args.target):
        files = sorted(f for f in glob.glob(os.path.join(args.target, "case_*.nii.gz"))
                       if not f.endswith("_i16.nii.gz"))
    else:
        files = [args.target]
    if not files:
        print("nothing to convert")
        return
    for f in files:
        convert(f)
    print(f"converted {len(files)} files")


if __name__ == "__main__":
    sys.exit(main())
