# Quickstart: Paired Per-Image IoU Test

This guide validates the TESTE A CLI after implementation. It does not train either
model. Run commands from the repository root unless a command explicitly changes to the
Python project directory.

## Prerequisites

- Python environment used by the experiment, with the existing PyTorch, SciPy, NumPy,
  pandas, OpenCV, and Matplotlib dependencies installed.
- Readable base and augmented checkpoints.
- The canonical 800-image real test set, containing matching `images/` and `masks/`
  directories. Use `D:\dataset\tgs-salt\subset_split\test` on Windows or
  `/var/tmp/cym7/datasets/subset_split/test` on HPC. The Windows folder's 800 image IDs
  match `D:\dataset\tgs-salt\subset_split\test_set.csv` exactly. Do not use
  `Salt-Segmentation-UNet/output/test_paths.txt`; it is a different split.

## 1. Run focused unit tests

```powershell
Set-Location Salt-Segmentation-UNet
python -m unittest discover -s tests -p "test_paired_iou_test.py" -v
Set-Location ..
```

Expected: all metric, pair-validation, statistical edge-case, and CLI validation tests
pass.

## 2. Run TESTE A with the documented checkpoints

On Windows, use the verified canonical folder:

```powershell
$expectedIds = @(Import-Csv 'D:\dataset\tgs-salt\subset_split\test_set.csv' | ForEach-Object { $_.id } | Sort-Object -Unique)
$actualIds = @(Get-ChildItem 'D:\dataset\tgs-salt\subset_split\test\images' -File | ForEach-Object { $_.BaseName } | Sort-Object -Unique)
if ($expectedIds.Count -ne 800 -or $actualIds.Count -ne 800 -or (Compare-Object $expectedIds $actualIds).Count -ne 0) { throw 'Canonical test IDs do not match test_set.csv.' }

python Salt-Segmentation-UNet/paired_iou_test.py `
  --base-checkpoint resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth `
  --aug-checkpoint resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth `
  --test-dir D:\dataset\tgs-salt\subset_split\test `
  --output-dir resultsHPC/paired_iou_test_a
```

On HPC, use the equivalent canonical path:

```bash
python Salt-Segmentation-UNet/paired_iou_test.py \
  --base-checkpoint resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth \
  --aug-checkpoint resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth \
  --test-dir /var/tmp/cym7/datasets/subset_split/test \
  --output-dir resultsHPC/paired_iou_test_a
```

Expected: the command loads both state dictionaries without training, validates the
same 800 canonical image/mask IDs, evaluates both models in batches with grayscale
128x128 inputs, and reports per-image IoU and Dice plus a one-sided paired Wilcoxon test
with threshold 0.5 and alpha 0.05. On Windows, verify that the IDs in `test/images`
match `test_set.csv` before the run.

## 3. Check outputs

Expected files under `resultsHPC/paired_iou_test_a/`:

- `per_image_metrics.csv`: exactly 800 ID-paired rows with base/augmented IoU and Dice,
  differences, and empty-ground-truth flags.
- `summary.json`: the test statistic, p-value, alpha, alternative, decision, effect size,
  diagnostics, input paths, effective settings, versions, and metric convention.
- `paired_iou.png` and `iou_difference.png`: paired scores and per-image difference
  distribution.

For exactly zero differences, expect W=0, p=1, and rank-biserial effect `null`; all
image rows remain in the CSV. An input/pairing error must return a nonzero exit status
and must not leave a success summary.

## 4. Verify CLI overrides

```powershell
python Salt-Segmentation-UNet/paired_iou_test.py --help
```

Rerun with alternate checkpoint, test, or output paths and explicit `--threshold`,
`--alpha`, `--alternative`, `--batch-size`, or `--device` values. Expected: the run
uses and records each supplied value; an invalid value is rejected before inference.
