# Data Model: Paired Per-Image IoU Test

The analysis is a stateless CLI run. It reads two checkpoints and one test dataset, then
writes image-level rows, a statistical summary, and diagnostic plots. There is no
persistent database or mutable workflow state.

## Test Pair

One canonical real test example shared by both model evaluations. The canonical
directory contains 800 image/mask pairs whose image IDs match `test_set.csv` exactly;
`output/test_paths.txt` is a different split and is not an input to this feature.

| Field | Type | Validation |
|---|---|---|
| `image_id` | string | Unique file stem; must exist in both image and mask sets |
| `image_path` | filesystem path | Readable grayscale image |
| `mask_path` | filesystem path | Readable binary ground-truth mask; resized with nearest neighbor |

**Canonical directory**: `/var/tmp/cym7/datasets/subset_split/test` on HPC or
`D:\dataset\tgs-salt\subset_split\test` on Windows.

**Relationship**: Each `Test Pair` yields exactly one `Per-Image Result` containing
metrics from both checkpoints.

## Per-Image Result

| Field | Type | Validation |
|---|---|---|
| `image_id` | string | Unique and stable; output rows are sorted by ID |
| `iou_base` | float in [0, 1] | Real-only checkpoint; computed per image |
| `iou_augmented` | float in [0, 1] | Real-plus-synthetic checkpoint; computed on the same image and mask |
| `iou_difference` | float in [-1, 1] | `iou_augmented - iou_base` |
| `dice_base` | float in [0, 1] | Descriptive metric for the real-only checkpoint |
| `dice_augmented` | float in [0, 1] | Descriptive metric for the augmented checkpoint |
| `dice_difference` | float in [-1, 1] | `dice_augmented - dice_base` |
| `ground_truth_empty` | boolean | True when the resized binary mask has no positive pixels |

**Relationship**: Exactly one result per validated `Test Pair`; no partial pairing is
written as a successful run.

## Paired Analysis Summary

One record for the entire CLI execution.

| Field group | Contents |
|---|---|
| Run inputs | Base and augmented checkpoint paths; test and output directories; evaluated image count and ID manifest or digest |
| Effective settings | Threshold, alpha, alternative, batch size, device, model input size, empty-union IoU convention |
| Environment | Python, PyTorch, NumPy, and SciPy versions |
| Descriptive metrics | Median and IQR for base/augmented IoU and Dice; median IoU difference; favorable-pair percentage; zero-difference count |
| Wilcoxon result | Alternative, statistic, p-value, alpha, decision, nonzero pair count, positive and negative rank sums |
| Effect size | Matched-pairs rank-biserial correlation; null when there are no nonzero differences |
| Diagnostics | Shapiro-Wilk statistic/p-value, skewness, or an explicit unavailable reason |
| Data integrity | Validated pair count, empty-ground-truth count, and any input validation errors |

## Output Files

- `per_image_metrics.csv`: one stable-ID-sorted row per `Test Pair`, including all fields
  from `Per-Image Result`.
- `summary.json`: one `Paired Analysis Summary`, including the effective configuration
  and metric convention.
- `paired_iou.png`: paired comparison of base and augmented IoU by image.
- `iou_difference.png`: distribution of `iou_difference` with a zero reference.

A successful run requires all test IDs to have exactly one readable image and mask and
requires both checkpoints to load into the same expected U-Net architecture. Invalid
inputs stop before a success summary is written.
