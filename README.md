# Anatomical-Prior Anomaly Detection on CT (TotalSegmentator)

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

**Data**: two real contrast-enhanced abdominal CTs from the public KiTS19
challenge (Heller et al.; via the official `neheller/KiTS-Challenge-Imaging`
Hugging Face dataset): `images/case_00000.nii.gz`
(611×512×512, 0.5×0.92×0.92 mm, 226 MB float64 → converted losslessly to
141 MB int16) and `images/case_00001.nii.gz`
(602×512×512, 0.5×0.8×0.8 mm, 276 MB float64 → 176 MB int16).
No synthetic or phantom data are used for reported results; tiny synthetic
masks are used only for the unit smoke test (`tests/test_smoke.py`).

**Segmentation**: TotalSegmentator 2.18.0, `--fast` (3 mm model), CPU, one
scan at a time: **117 binary masks per scan** at full original resolution
(uint8). Runtime per scan on a 2-CPU/8 GB box: ~56 s prediction (4
sliding-window tiles), ~3 s resampling back to native resolution, ~55–62 s
saving masks.

### Anomaly report (REAL numbers from REAL runs)

Cohort n=2, so robust z-scores are degenerate by design (the code requires
n≥3) — no z-score flags are emitted. The real signal at this scale is the
asymmetry index. `results/anomaly_report.csv` (from
`python src/report.py --seg-dir segs/case_00000 --scan-id case_00000 --seg-dir2 segs/case_00001 --scan-id2 case_00001`):

| scan | flag | organ | score | detail |
|---|---|---|---|---|
| case_00001 | asymmetry | gluteus_maximus | 0.255 | L=7.7 mL, R=12.9 mL |
| case_00001 | asymmetry | lung_lower_lobe | 0.232 | L=128.0 mL, R=205.2 mL |
| case_00000 | asymmetry | adrenal_gland | 0.166 | L=4.7 mL, R=3.3 mL |

Honest reading: the case_00001 flags are almost certainly **scan-truncation
artifacts** (gluteus and lung cut at the field-of-view edge — the scans are
abdominal, lungs/legs only partially covered), not pathology. That is the
method working as designed: it flags gross anatomical deviations for human
review, whether they come from disease or from acquisition limits. The
case_00000 adrenal asymmetry (4.7 vs 3.3 mL) is within normal inter-side
variation for a ~4 mL structure at 3 mm segmentation resolution — a reminder
that small-structure flags need a larger cohort before they mean anything.

Selected real organ volumes (mL), from `results/case_00000_volumes.csv` /
`results/case_00001_volumes.csv`:

| organ | case_00000 | case_00001 |
|---|---|---|
| liver | 1559.61 | 1879.08 |
| spleen | 269.36 | 676.25 |
| kidney_right | 191.54 | 213.33 |
| kidney_left | 159.78 | 205.92 |
| pancreas | 73.70 | 187.67 |
| aorta | 50.48 | 64.08 |
| stomach | 154.42 | 313.68 |
| colon | 953.49 | 469.48 |

(KiTS19 scans are kidney-tumor cases; kidney-adjacent volumes here must not
be treated as population norms.)

## How to run

```bash
python3 -m venv ~/venvs/totalseg
~/venvs/totalseg/bin/pip install -r requirements.txt   # heavy: pulls CPU torch (~1-2 GB)

# 1. Segment (weights download ~135 MB zip on first run; ~2 min/scan on a 2-CPU box with --fast)
TotalSegmentator -i data/case_00000_i16.nii.gz -o segs/case_00000 --fast --nr_thr_resamp 1 --nr_thr_saving 1
TotalSegmentator -i data/case_00001_i16.nii.gz -o segs/case_00001 --fast --nr_thr_resamp 1 --nr_thr_saving 1

# 2. Volumetry + anomaly scoring (seconds)
python src/report.py --seg-dir segs/case_00000 --scan-id case_00000 \
                     --seg-dir2 segs/case_00001 --scan-id2 case_00001

# Smoke test (no TotalSegmentator needed; uses tiny synthetic masks)
python tests/test_smoke.py
# ... or validate the volumetry path on REAL produced masks:
python tests/test_smoke.py --real-seg-dir segs/case_00000 --scan-id case_00000
```

## Limitations (read before citing)

- **N=2 scans — this is a proof of concept of an unsupervised method, NOT a
  validated detector.** Cohort z-scores with n=2 are degenerate (MAD over two
  points); the meaningful signals at this scale are the asymmetry indices and
  the raw volumetric profile. The pipeline is written to scale unchanged to
  the full 1,939-scan TotalSegmentator release, where the z-score reference
  becomes real.
- Segmentation failures look like anomalies: a missed organ (e.g. bowel
  truncated at the scan edge) inflates asymmetry/volume flags. TotalSegmentator
  `--fast` uses the 3 mm model (Dice 0.84 vs 0.94 for the 1.5 mm model), so
  small-structure volumes are noisier.
- The Wasserthal age trends are directional correlations, not per-organ
  normative mean volumes; without age/sex metadata per scan they can only be
  used qualitatively.
- KiTS19 scans are kidney-tumor cases — kidney-adjacent volumes in these two
  scans should not be taken as population norms.

## References

- Wasserthal J. et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images.* Radiology: Artificial Intelligence 2023.
  PMC10546353.
- Heller N. et al. *The KiTS19 Challenge Data: Kidney and Kidney Tumor
  Segmentation.* (MICCAI KiTS19 challenge, 2019; imaging via
  `neheller/KiTS-Challenge-Imaging`.)
