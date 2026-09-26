"""Fast smoke test for volumetry + anomaly scoring.

Builds tiny synthetic binary masks (no TotalSegmentator needed), computes
volumes, and runs the anomaly scorer end to end. A second mode runs the same
pipeline on REAL produced masks if --real-seg-dir is given.

Usage:
    python tests/test_smoke.py
    python tests/test_smoke.py --real-seg-dir /path/to/totalseg/output --scan-id kits19_00000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nibabel as nib
from volumetry import scan_volumes, mask_volume
from anomaly_scoring import load_volumes, cohort_z_flags, asymmetry_flags, asymmetry_index


def make_mask(path: str, shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0), frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) < frac).astype(np.uint8)
    img = nib.Nifti1Image(data, np.diag([*spacing, 1.0]))
    nib.save(img, path)


def test_volumetry():
    with tempfile.TemporaryDirectory() as d:
        make_mask(os.path.join(d, "liver.nii.gz"), spacing=(2.0, 2.0, 2.0))
        v = mask_volume(os.path.join(d, "liver.nii.gz"))
        # voxel volume = 8 mm^3 = 0.008 mL
        assert abs(v.volume_ml - v.voxels * 0.008) < 1e-6, v
        vols = scan_volumes(d)
        assert len(vols) == 1 and vols[0].name == "liver"
    print("volumetry OK")


def test_asymmetry():
    assert abs(asymmetry_index(100.0, 100.0)) < 1e-9
    assert asymmetry_index(150.0, 100.0) > asymmetry_index(110.0, 100.0)
    with tempfile.TemporaryDirectory() as d:
        make_mask(os.path.join(d, "kidney_left.nii.gz"), frac=0.5, seed=1)
        make_mask(os.path.join(d, "kidney_right.nii.gz"), frac=0.01, seed=2)  # tiny -> asymmetric
        csvp = os.path.join(d, "vol.csv")
        from volumetry import write_csv
        write_csv(scan_volumes(d), csvp)
        scans = load_volumes([csvp])
        # csv scan id is "vol"; asymmetry should fire for kidney
        flags = asymmetry_flags(scans)
        assert any(f.organ == "kidney" for f in flags), flags
    print("asymmetry OK")


def test_cohort_z():
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, frac in enumerate([0.1, 0.1, 0.1, 0.1, 0.9]):  # last is outlier
            sd = os.path.join(d, f"scan{i}")
            os.makedirs(sd)
            make_mask(os.path.join(sd, "liver.nii.gz"), frac=frac, seed=i)
            csvp = os.path.join(d, f"s{i}.csv")
            from volumetry import write_csv
            write_csv(scan_volumes(sd), csvp)
            paths.append(csvp)
        scans = load_volumes(paths)
        flags = cohort_z_flags(scans, z_thresh=3.0)
        assert any("liver" in f.organ for f in flags), "expected liver outlier flag"
    print("cohort z-score OK")


def test_cohort_summary():
    from cohort_summary import cohort_stats, markdown_table
    with tempfile.TemporaryDirectory() as d:
        paths = []
        # 4 scans, liver volumes exactly 100,200,300,400 mL -> median 250, Q1 175, Q3 325
        for i, vol_ml in enumerate([100.0, 200.0, 300.0, 400.0]):
            csvp = os.path.join(d, f"s{i}.csv")
            with open(csvp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["organ", "voxels", "volume_ml"])
                w.writerow(["liver", 1000, f"{vol_ml:.2f}"])
                w.writerow(["kidney_left", 100, "10.00"])
            paths.append(csvp)
        stats = cohort_stats(paths)
        liver = next(s for s in stats if s.organ == "liver")
        assert liver.n == 4
        assert abs(liver.median_ml - 250.0) < 1e-6, liver
        assert abs(liver.q1_ml - 175.0) < 1e-6, liver
        assert abs(liver.q3_ml - 325.0) < 1e-6, liver
        kidney = next(s for s in stats if s.organ == "kidney_left")
        assert abs(kidney.median_ml - 10.0) < 1e-6
        md = markdown_table(stats, ["liver", "missing_organ"])
        assert "| liver | 4 | 250.0 |" in md and "| missing_organ | 0 |" in md
    print("cohort summary OK")


def test_flag_recurrence():
    from anomaly_scoring import Flag, flag_recurrence
    flags = [
        Flag("s1", "asymmetry", "kidney", 0.5, 0.0),
        Flag("s2", "asymmetry", "kidney", 0.4, 0.0),
        Flag("s1", "volume_outlier", "liver", 3.5, 100.0),
    ]
    rec = flag_recurrence(flags)
    assert rec[0][:3] == ("asymmetry", "kidney", 2), rec
    assert rec[0][3] == ["s1", "s2"]
    assert rec[1][:3] == ("volume_outlier", "liver", 1)
    print("flag recurrence OK")


def test_real_masks(seg_dir: str, scan_id: str):
    from volumetry import write_csv
    vols = scan_volumes(seg_dir)
    print(f"{scan_id}: {len(vols)} masks -> volumes computed")
    assert len(vols) >= 50, f"expected >=50 masks from TotalSegmentator, got {len(vols)}"
    with tempfile.TemporaryDirectory() as d:
        csvp = os.path.join(d, "vol.csv")
        write_csv(vols, csvp)
        scans = {scan_id: scans} if False else load_volumes([csvp])
    for r in sorted(vols, key=lambda x: -x.volume_ml)[:10]:
        print(f"  {r.name}: {r.volume_ml:.1f} mL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-seg-dir", help="Real TotalSegmentator output dir")
    ap.add_argument("--scan-id", default="scan1")
    args = ap.parse_args()

    test_volumetry()
    test_asymmetry()
    test_cohort_z()
    test_cohort_summary()
    test_flag_recurrence()
    if args.real_seg_dir:
        test_real_masks(args.real_seg_dir, args.scan_id)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
