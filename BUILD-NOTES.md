# Build notes — honest log of what happened

## Data source (final)
- **Dataset**: KiTS19 (kidney tumor segmentation challenge, Heller et al. 2019),
  via the official public Hugging Face dataset `neheller/KiTS-Challenge-Imaging`.
- **Files** (individual per-case NIfTI, direct download):
  - `https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_00000.nii.gz`
  - `https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_00001.nii.gz`
- **Verified on 2026-09-22** (gzip integrity OK, headers read with nibabel):
  - `case_00000`: 225,959,569 bytes → shape (611, 512, 512), spacing [0.5, 0.92, 0.92] mm
  - `case_00001`: 276,387,358 bytes → shape (602, 512, 512), spacing [0.5, 0.8, 0.8] mm
  - Both stored as **float64** (unusual for CT; HU values are integers in [-1024, 1413]).
  - Converted losslessly to int16 (`case_00000_i16.nii.gz` 140,626,644 bytes;
    `case_00001_i16.nii.gz` 175,784,068 bytes; range check asserted before cast).
    Originals deleted after conversion. This is a dtype cast of the real scan,
    not a substitution.
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

## Anomaly pipeline on real masks (2026-09-22)
- `python src/report.py --seg-dir segs/case_00000 --scan-id case_00000
  --seg-dir2 segs/case_00001 --scan-id2 case_00001`
  → `results/anomaly_report.csv` (gitignored; key numbers copied into README).
- 61 / 59 organ volumes per scan. 3 asymmetry flags, 0 z-score flags
  (correctly suppressed: robust z needs n≥3).
- Flags: case_00001 gluteus_maximus 0.255 (L=7.7/R=12.9 mL), case_00001
  lung_lower_lobe 0.232 (L=128.0/R=205.2 mL), case_00000 adrenal_gland 0.166
  (L=4.7/R=3.3 mL). The case_00001 flags are field-of-view truncation, not
  pathology — noted honestly in the README.

## Method honesty
- N=2: cohort z-scores are degenerate at this scale (the code refuses n<3);
  the anomaly report's meaningful content at this scale is the asymmetry
  indices + raw organ volumes. Stated up front in README.
- KiTS19 cases are kidney-tumor scans, so kidney volumes/asymmetry here are
  NOT population norms — stated in README limitations.
- The Wasserthal et al. 2023 age trends are directional Spearman
  correlations, not normative mean/SD tables; the code and README call them
  trend priors and never fabricate z-scores from them.
- The venv patches above are environment workarounds for a 2-CPU/8 GB box,
  not part of the method; the repo's own `src/` code is original and
  unpatched. Anyone rerunning on a bigger machine can use the stock
  TotalSegmentator install.
