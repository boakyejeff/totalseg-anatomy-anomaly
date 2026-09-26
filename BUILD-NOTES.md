# Build notes — honest log of what happened

## Data source (final): 12 scans
- **Dataset**: KiTS19 (kidney tumor segmentation challenge, Heller et al. 2019),
  via the official public Hugging Face dataset `neheller/KiTS-Challenge-Imaging`
  (dataset SHA `65f1f295873a326230153c7e1de0c7dba10f0b29`, verified via API).
- **Files** (individual per-case NIfTI, direct download):
  - `https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_00000.nii.gz`
    through `.../case_00011.nii.gz` (12 scans).
- **Verified 2026-09-22** (case_00000, case_00001) **and 2026-09-26**
  (case_00002–case_00011): gzip integrity OK (`gzip -t`), headers read with
  nibabel. All stored as **float64** (unusual for CT; HU integers in
  [-1024, 3071]); converted losslessly to int16 with range-check + round-trip
  verification before deleting originals. Per-scan facts:

  | case | shape | HU range | int16 size |
  |---|---|---|---|
  | 00000 | 611×512×512 | −1024…1413 | 141 MB |
  | 00001 | 602×512×512 | −1024…1413 | 176 MB |
  | 00002 | 261×512×512 | −1024…3071 | 63 MB |
  | 00003 | (see logs) | −1024…3071 | — |
  | 00004 | (see logs) | — | — |
  | 00005 | 834×512×512 | −1024…3071 | 198 MB (slab-wise memmap conversion; whole-array did not fit RAM) |
  | 00006 | 157×512×512 | −1024…1782 | 50 MB |
  | 00007 | 61×512×512 | −1024…1088 | 17 MB |
  | 00008 | 227×512×512 | −1024…3071 | 68 MB |
  | 00009 | 77×512×512 | −1024…1374 | 23 MB |
  | 00010 | 50×512×512 | −1024…1280 | 16 MB |
  | 00011 | 80×512×512 | −1024…1734 | 27 MB |

  (A first conversion attempt reused the float64 NIfTI header and silently
  produced float64 output; caught by dtype verification, deleted, redone with
  the header dtype set to int16. `src/convert_dtype.py` is whole-array and
  unsuitable for very large scans on small boxes — noted in code.)
- **Why not LUNA16**: `s3://luna16` denies unsigned LIST and unsigned GET
  (AccessDenied on both `aws`-style listing and direct HTTPS object GET),
  and the AWS CLI is not installed here. The task brief allowed falling back
  to another public per-file CT source; KiTS19 abdominal contrast-enhanced CTs
  are real clinical scans and are individually downloadable.
- All scans are REAL patient CTs. No synthetic/phantom data are used for
  reported results; tiny synthetic masks are used only for the unit smoke test.

## Environment
- VM: 2 vCPU, ~7.7 GB RAM (only ~2.5–4.9 GB actually available at runtime),
  100 GB disk, no swap, no GPU.
- Python 3.12.3, venv at `~/venvs/totalseg` (kept OUTSIDE the repo; gitignored).
- `pip install` was painful and is worth recording:
  - TotalSegmentator 2.18.0 + torch 2.14.0+cpu + torchvision 0.29.0+cpu +
    SimpleITK, nibabel, pandas, scipy. Venv ≈ 2.5 GB.
  - Large downloads kept failing mid-stream (flaky); fixed with
    `TMPDIR=~/tmp_pip --no-cache-dir` and resume retries.
  - PyPI `torchvision==0.29.0` raised `operator torchvision::nms does not
    exist` against CPU torch; force-reinstalled `torchvision==0.29.0+cpu`
    from the PyTorch CPU index.
  - First-run weight auto-download stalled; manually fetched
    `https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip`
    (135,386,075 bytes, `ZipFile.testzip()` clean) into
    `~/.totalsegmentator/nnunet/results/.../fold_0/checkpoint_final.pth`
    (164,939,235 bytes).

## TotalSegmentator runs — the hard part (all on 2026-09-22)
Getting inference to survive on this box took several rounds of real
debugging; nothing was faked or skipped:

1. **SIGKILL at ~30 s during image load.** Root cause: `nnunet.py` does
   `nib.Nifti1Image(img_in_orig.get_fdata(), ...)` — float64, so the
   160 M-voxel scan materialized twice (~2.6 GB) on top of torch (~1 GB)
   with only ~2.9 GB available. **Fix (venv patch, documented):**
   `get_fdata(dtype=np.float32)`, halving it. Backup of the original kept at
   `nnunet.py.bak`.
2. **nnU-Net preprocessing workers died with `ConnectionRefusedError`.**
   Misleading symptom: with 1 preprocessing worker the single case was
   preprocessed fine, but the main process never consumed the queue — it was
   dying in sliding-window inference (see 4), which tore down the manager.
3. **`predict_from_files` → `predict_from_files_sequential`.** The
   multiprocessing manager path is fragile here; nnU-Net ships a
   documented sequential fallback ("Slow, but sometimes necessary").
   **Fix (venv patch):** call it instead. No behavior change to the math.
4. **Silent death at first sliding-window tile (0/4) with RAM free.**
   An isolated forward pass worked (13.6 s, 2.2 GB peak). A full 4-tile run
   with mirroring worked too (3.7 GB peak). The pipeline additionally holds
   the full-res image, and box memory fluctuates (~1 GB swings from other
   processes), so the pipeline OOMed intermittently. `swapon` is blocked in
   this container (Operation not permitted) despite root.
   **Fix (venv patch):** after the 3 mm resampled input is written to tmp,
   stash only `affine`/`shape`/`header` metadata in a tiny `_OrigImageMeta`
   shim and `del` the full-res arrays + `gc.collect()` before prediction
   (~1.3 GB freed). The resample-back step only ever touches
   `img_in.affine` (canonical), and `undo_canonical`/shape-check only touch
   `.affine`/`.shape`/`.header` — verified in source before patching.
   Backup at `nnunet.py.bak2`.
5. **Success.** Both scans segmented end-to-end:
   - `TotalSegmentator -i data/case_00000_i16.nii.gz -o segs/case_00000 --fast --nr_thr_resamp 1 --nr_thr_saving 1`
   - case_00000: predicted in 55.93 s (4/4 tiles @ ~12–18 s/it), resampled
     3.01 s, saved 61.74 s → **117 masks**, uint8, (611, 512, 512).
   - case_00001: same command → **117 masks**, uint8, (602, 512, 512)
     (predicted ~53 s, resampled 3.13 s, saved 54.85 s).
   - The current `--fast` total model emits 117 masks (the 2023 paper's 104
     refers to the original release).
   - Full logs: `logs/scan1.log`, `logs/scan2.log` (gitignored).

## Scale-up to 12 scans (2026-09-26) — OOM retries
The box is shared with other agents' training/registration jobs, so available
RAM fluctuated between ~0.4 and ~4.5 GB. Inference survived, but several scans
were OOM-killed (exit 137) right after the prediction tiles, with 0 masks
saved — the post-prediction resample-back and save phases held full labelmaps
as float64. Two more venv-only patches (backup `nnunet.py.bak3`,
behavior-preserving — label values are integers either way):
(1) `save_segmentation_nifti`: `img.get_fdata()` → `np.asarray(img.dataobj)`
(avoided one full-labelmap float64 load per each of 117 classes);
(2) resample-back: `img_pred.get_fdata().astype(np.uint8)` →
`np.asarray(img_pred.dataobj).astype(np.uint8)`.
Retry pattern: sequential loop, skip dirs already having ≥100 masks, plus a
memory gate (wait until ≥3 GB available before launching; 15-min cap, then
proceed anyway). Failed scans were retried on later passes when the box was
quieter — OOM kills were intermittent timing luck, not scan-specific.

Final successful wall times (exit 0, 117 masks each):
- case_00002: 174 s; case_00003: 271 s; case_00004: 78 s; case_00005: 198 s;
  case_00006: 150 s; case_00007: 134 s; case_00008: 293 s; case_00009: 132 s;
  case_00010: 129 s; case_00011: 131 s
  (plus case_00000: ~121 s and case_00001: ~111 s from 2026-09-22).
All 12 scans: 117/117 masks, uint8, at native resolution.

## Anomaly pipeline on real masks (2026-09-22, n=2)
- `python src/report.py --seg-dir segs/case_00000 --scan-id case_00000
  --seg-dir2 segs/case_00001 --scan-id2 case_00001`
  → `results/anomaly_report.csv` (gitignored; key numbers copied into README).
- 61 / 59 organ volumes per scan. 3 asymmetry flags, 0 z-score flags
  (correctly suppressed: robust z needs n≥3).
- Flags: case_00001 gluteus_maximus 0.255 (L=7.7/R=12.9 mL), case_00001
  lung_lower_lobe 0.232 (L=128.0/R=205.2 mL), case_00000 adrenal_gland 0.166
  (L=4.7/R=3.3 mL). The case_00001 flags are field-of-view truncation, not
  pathology — noted honestly in the README.

## Anomaly pipeline, full cohort (2026-09-26, n=12)
- `python src/volumetry.py --seg-dir segs/case_XXXXX --out
  results/case_XXXXX_volumes.csv` → 12 per-scan CSVs (45–109 organs/scan;
  fewer for truncated FOVs, e.g. case_00007: 46, case_00010: 45).
- `python src/report.py --volumes-csv results/case_*_volumes.csv --out
  results/anomaly_report.csv --asymmetry-out results/asymmetry_indices.csv`
- `python src/cohort_summary.py results/case_*_volumes.csv --out
  results/cohort_summary.csv`
- New code (2026-09-26): `all_asymmetry_indices()` /
  `write_asymmetry_csv()` in `src/anomaly_scoring.py` export the complete
  per-organ L/R asymmetry index for every scan (not only thresholded flags);
  `src/report.py` gained `--asymmetry-out` and repeatable `--scan DIR=ID`
  / `--volumes-csv`; `src/cohort_summary.py` gained README-ready Markdown
  tables; `tests/test_smoke.py` covers the new functions.
- Real results (n=12): kidney_left median 196.3 mL (IQR 166.3–229.4);
  kidney_right median 182.0 mL (IQR 160.3–214.9). Kidney asymmetry flagged in
  3/12: case_00005 AI=0.903 (L=186.6/R=9.6 mL), case_00008 AI=0.481
  (L=228.9/R=653.0 mL), case_00011 AI=0.155 (L=207.4/R=151.8 mL); other 9
  scans 0.002–0.117. `kidney_cyst_right` segmented in 3/12 (17.2–26.6 mL).
  Remaining flags are FOV-truncation artifacts (gluteals, iliac vessels, ribs,
  lung lobes at scan edges). No diagnostic claims: KiTS19 is a kidney-tumor
  cohort, no lesion ground truth was used, no sensitivity/specificity exists.

## Method honesty
- N=12 (was N=2 on 2026-09-22): cohort z-scores are computed (the code
  requires n≥3) but remain unstable at this scale — stated up front in README.
  The anomaly report's most interpretable content is the asymmetry indices +
  raw organ volumes. The pipeline scales unchanged to thousands of scans.
- KiTS19 cases are kidney-tumor scans, so kidney volumes/asymmetry here are
  NOT population norms and no disease-detection performance is claimed —
  stated in README limitations. Unilateral kidney asymmetry is expected in
  this clinical setting; flags are for human review, not diagnosis.
- The Wasserthal et al. 2023 age trends are directional Spearman
  correlations, not normative mean/SD tables; the code and README call them
  trend priors and never fabricate z-scores from them.
- The venv patches above are environment workarounds for a 2-CPU/8 GB box,
  not part of the method; the repo's own `src/` code is original and
  unpatched. Anyone rerunning on a bigger machine can use the stock
  TotalSegmentator install.
