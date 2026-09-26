# Anatomical-Prior Anomaly Detection on CT (TotalSegmentator)

![Project cover](assets/cover.png)
> **NOVELTY CLAIM.** Unsupervised anomaly detection on CT that uses 100+
> whole-body anatomical structures — segmented automatically by
> TotalSegmentator — as *structural priors*, with **zero disease labels**.
> Instead of training a pathology classifier, we flag per-organ volume
> outliers (robust z-scores against the cohort) and left–right asymmetry of
> paired organs, calibrated against published age–volume trends from the
> TotalSegmentator aging analysis (Wasserthal et al., *Radiology: Artificial
> Intelligence* 2023, n=4,004). No lesion annotations, no abnormality labels,
> no supervision of any kind on disease.

## Problem

Radiology AI anomaly detection usually needs labeled pathology (tumors,
lesions) or trains a one-class model on raw voxels. Both are label-hungry or
opaque. Here we ask: *how far can you get by treating accurate anatomical
segmentation itself as the feature space?* If a model can reliably outline
104+ structures (117 masks with the current --fast model), then gross anatomical deviations — a shrunken kidney, a
lopsided lung, an aneurysmal aorta — become detectable by simple,
interpretable statistics over organ volumes, with no disease labels at all.

## Method

1. **Segment** a CT with TotalSegmentator (`--fast`, CPU): 117 structures →
   one binary mask per class (the current `--fast` total model emits 117
   masks; the 2023 paper reported 104 for the original release).
2. **Volumetry** (`src/volumetry.py`): per-organ volumes in mL
   (voxel count × voxel volume from NIfTI spacing).
3. **Anomaly scoring** (`src/anomaly_scoring.py`):
   - *Cohort volume z-scores*: robust median/MAD z per organ across the
     cohort — flags organs that are volumetrically unusual.
   - *Asymmetry index*: `|L−R|/(L+R)` for paired organs (kidneys, lung lobes,
     hips, gluteals, …); flags ≥ 0.15. Normal anatomy is nearly symmetric, so
     strong asymmetry flags unilateral pathology *or* segmentation failure,
     both of which a human should review.
   - *Age-trend priors* from Wasserthal et al. 2023 (kidney rs=−0.49, pancreas
     rs=−0.49, aorta rs=+0.64, iliac arteries rs=+0.33, iliopsoas rs=−0.61):
     directional sanity checks only — the scans used here carry no age
     metadata.
4. **Report** (`src/report.py`): ranked per-scan flag list with scores.

## Real results

**Data**: twelve real contrast-enhanced abdominal CTs from the public KiTS19
challenge (Heller et al.; via the official `neheller/KiTS-Challenge-Imaging`
Hugging Face dataset): `case_00000` through `case_00011`. All scans were
converted losslessly from float64 to int16 (verified value-range and
round-trip; e.g. `case_00005`: 834×512×512, range −1024…3071, 313 MB → 198 MB).
No synthetic or phantom data are used for reported results; tiny synthetic
masks are used only for the unit smoke test (`tests/test_smoke.py`).

**Cohort caveat (important).** KiTS19 is a *kidney-tumor clinical cohort* —
these are not healthy controls. Unilateral kidney asymmetry is *expected* in
this setting (tumors, cysts, prior nephrectomy, atrophy). The asymmetry flags
below are anatomical observations, not diagnoses: this pipeline has no disease
labels, no radiologist ground truth, and makes no disease-detection claims.
Kidney volumes here must not be treated as population norms.

**Segmentation**: TotalSegmentator 2.18.0, `--fast` (3 mm model), CPU, one
scan at a time: **117 binary masks per scan** at full original resolution
(uint8) — 1,404 masks total. Wall time per scan on a shared 2-CPU/8 GB box:
78–293 s (typical 130–175 s: ~20–35 s/tile prediction, ~3 s resampling,
~60 s saving). Several scans needed OOM retries (see Limitations).

### Anomaly report (REAL numbers from REAL runs, n=12)

`results/anomaly_report.csv` (from `python src/report.py --volumes-csv
results/case_*_volumes.csv`), `results/cohort_summary.csv` (per-organ median
+ IQR), and `results/asymmetry_indices.csv` (complete L/R index per paired
organ per scan — not just thresholded flags).

Cohort organ volumes, median (Q1–Q3), mL:

| organ | n | median | Q1 | Q3 |
|---|---|---|---|---|
| liver | 12 | 1941.3 | 1541.5 | 2226.7 |
| colon | 12 | 634.2 | 468.4 | 790.1 |
| spleen | 12 | 275.4 | 152.1 | 319.2 |
| stomach | 12 | 269.2 | 163.1 | 428.1 |
| kidney_left | 12 | 196.3 | 166.3 | 229.4 |
| kidney_right | 12 | 182.0 | 160.3 | 214.9 |
| pancreas | 12 | 88.2 | 78.9 | 98.8 |
| aorta | 12 | 60.8 | 52.0 | 109.7 |

Top kidney asymmetry flags (index `|L−R|/(L+R)`, warn ≥ 0.15):

| scan | L (mL) | R (mL) | index |
|---|---|---|---|
| case_00005 | 186.6 | 9.6 | 0.903 |
| case_00008 | 228.9 | 653.0 | 0.481 |
| case_00011 | 207.4 | 151.8 | 0.155 |

The other 9 scans have kidney asymmetry indices 0.002–0.117 (near-symmetric).
Robust z-scores also flag `kidney_right` as a volume outlier in case_00005
(9.6 mL vs cohort median 182.0 mL, z=−3.78) and case_00008 (653.0 mL vs
182.0 mL, z=+10.32). TotalSegmentator's `kidney_cyst_right` class fired in
3/12 scans (17.2–26.6 mL).

Honest reading: in a kidney-tumor cohort, these are the expected unilateral
kidney deviations — a near-absent right kidney (case_00005), a markedly
enlarged right kidney (case_00008), and a mild asymmetry (case_00011). The
pipeline flags them for human review; it does **not** diagnose tumors, cysts,
or nephrectomy — there is no lesion ground truth here and no
disease-detection performance is claimed.

Other recurring flags are acquisition/truncation artifacts, not pathology:
gluteal/iliopsoas/iliac-vessel asymmetries at the field-of-view edge
(e.g. case_00005 gluteus_medius L=13.2 vs R=0.8 mL; case_00007 iliac_vena
L=0.1 vs R=0.6 mL — sub-mL structures at 3 mm resolution), rib outliers from
partial coverage, and lung-lobe asymmetry where the abdominal FOV clips the
lungs. Small-structure flags need a larger cohort before they mean anything.

## How to run

```bash
python3 -m venv ~/venvs/totalseg
~/venvs/totalseg/bin/pip install -r requirements.txt   # heavy: pulls CPU torch (~1-2 GB)

# 1. Segment (weights download ~135 MB zip on first run; ~2 min/scan on a 2-CPU box with --fast)
TotalSegmentator -i data/case_00000_i16.nii.gz -o segs/case_00000 --fast --nr_thr_resamp 1 --nr_thr_saving 1
TotalSegmentator -i data/case_00001_i16.nii.gz -o segs/case_00001 --fast --nr_thr_resamp 1 --nr_thr_saving 1

# 2. Volumetry + anomaly scoring (seconds; repeat --scan per scan, or use
#    precomputed --volumes-csv files)
python src/report.py --scan segs/case_00000=case_00000 \
                     --scan segs/case_00001=case_00001 \
                     --out results/anomaly_report.csv
#    (also writes results/asymmetry_indices.csv: every paired organ's L/R
#     index per scan, not just thresholded flags)

# Cohort median/IQR table (README-ready):
python src/cohort_summary.py results/case_*_volumes.csv --out results/cohort_summary.csv

# Smoke test (no TotalSegmentator needed; uses tiny synthetic masks)
python tests/test_smoke.py
# ... or validate the volumetry path on REAL produced masks:
python tests/test_smoke.py --real-seg-dir segs/case_00000 --scan-id case_00000
```

## Limitations (read before citing)

- **N=12 scans from a kidney-tumor clinical cohort — this is a proof of
  concept of an unsupervised method, NOT a validated detector.** Robust
  z-scores are computed (n≥3 required by the code) but remain unstable at
  n=12; treat them as exploratory. The pipeline is written to scale unchanged
  to the full 1,939-scan TotalSegmentator release, where the z-score
  reference becomes real.
- **No disease-detection claims.** KiTS19 cases have kidney tumors, but this
  repo has no lesion annotations and no radiologist review. Asymmetry/volume
  flags are anatomical observations for human review, not diagnoses, and no
  sensitivity/specificity is reported (none can be computed).
- Segmentation failures look like anomalies: a missed organ (e.g. bowel
  truncated at the scan edge) inflates asymmetry/volume flags. TotalSegmentator
  `--fast` uses the 3 mm model (Dice 0.84 vs 0.94 for the 1.5 mm model), so
  small-structure volumes are noisier.
- The Wasserthal age trends are directional correlations, not per-organ
  normative mean volumes; without age/sex metadata per scan they can only be
  used qualitatively.
- **Low-RAM reality.** On the shared 2-CPU/8 GB box used here, TotalSegmentator
  runs were OOM-killed intermittently (exit 137) by co-tenant training jobs;
  a memory-gated retry loop (wait for ≥3 GB free, skip completed scans)
  eventually completed all 12. Two float64 transients in the installed
  TotalSegmentator were patched in the *environment only* (never in this
  repo) — see BUILD-NOTES.md. Runtimes above are wall-clock on that box.

## References

- Wasserthal J. et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images.* Radiology: Artificial Intelligence 2023.
  PMC10546353.
- Heller N. et al. *The KiTS19 Challenge Data: Kidney and Kidney Tumor
  Segmentation.* (MICCAI KiTS19 challenge, 2019; imaging via
  `neheller/KiTS-Challenge-Imaging`.)
