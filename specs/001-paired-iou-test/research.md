# Research: Paired Per-Image IoU Test

## Decision 1: Keep a standalone command-line script

**Decision**: Add `Salt-Segmentation-UNet/paired_iou_test.py`; do not extend or replace
`evaluate.py`.

**Rationale**: The existing evaluator groups whole-run metrics by scenario and seed,
then applies a paired t-test across runs. TESTE A pairs individual images and uses a
Wilcoxon signed-rank test. Sharing the current entry point could confuse the units of
analysis or change existing summaries. A single `argparse` script follows the project's
existing `train.py` and data utility style without adding a web layer or service.

**Alternatives considered**: Add a mode to `evaluate.py`; rejected because it mixes
run-level and image-level analyses in a tool that currently scans result CSV files.

## Decision 2: Reuse the current U-Net and preprocessing conventions

**Decision**: Instantiate `utils.model.UNet`, load each checkpoint as a state dict,
and use the current grayscale image, 128x128 resize, [0,1] tensor and nearest-neighbor
binary-mask behavior. Process both models in configurable batches on the same IDs.

**Rationale**: Both saved checkpoints come from the same project architecture, and
`train.py` defines these preprocessing and threshold conventions. Reusing these local
interfaces keeps the analysis comparable and avoids duplicating the network.

**Alternatives considered**: Load complete serialized model objects or copy the model
implementation into the new script; rejected because training saves state dicts and
`UNet` is already available in the project.

## Decision 3: Use the canonical test split and pair strictly by ID

**Decision**: Use `/var/tmp/cym7/datasets/subset_split/test` on HPC or
`D:\dataset\tgs-salt\subset_split\test` on Windows. The canonical folder must contain
the 800 IDs listed by `D:\dataset\tgs-salt\subset_split\test_set.csv`. Build
image/mask maps keyed by file stem, reject duplicate IDs or any missing image/mask, sort
validated IDs, and calculate both model results for every ID. Do not silently intersect
incomplete lists or substitute `Salt-Segmentation-UNet/output/test_paths.txt`.

**Rationale**: The local canonical test folder has 800 image and 800 mask files, and its
image IDs match `test_set.csv` exactly. The code-generated `output/test_paths.txt` also
has 800 entries but only 168 IDs overlap with the canonical CSV; equal counts do not
establish equal test sets. The paired test is valid only when both model scores refer to
the same canonical image and mask. Stable ID ordering makes CSV output reproducible.

## Decision 4: Compute metrics per image; test IoU only

**Decision**: Calculate per-image IoU and Dice from thresholded masks. Use IoU as the
sole primary Wilcoxon endpoint and report Dice descriptively. Calculate the paired
difference as `iou_augmented - iou_base`; use a one-sided `greater` alternative and
alpha 0.05 by default. Allow the CLI to set threshold, alpha, alternative, batch size,
and device, recording effective values in the summary.

**Rationale**: The study document defines TESTE A on per-image IoU and a directional
hypothesis. The constitution also requires IoU and Dice reporting, but does not require
a second inferential test for Dice.

**Alternatives considered**: Apply the Wilcoxon test to Dice too; rejected because it
adds an unrequested hypothesis and multiple-testing decision.

## Decision 5: Use spec-approved empty-union IoU convention

**Decision**: Preserve the approved spec and existing `train.py` metric convention:
`intersection / (union + 1e-8)`, which yields IoU 0 when the union is empty. Record this
convention in the output metadata and count test masks whose ground truth is empty.

**Rationale**: The spec adopted the training metric to preserve comparability. The
sample `compute_iou` in the TeX document instead returns IoU 1 when both prediction and
ground truth are empty. These conventions can produce a difference as large as 1 for
an empty-ground-truth image predicted empty by one model and positive by the other; the
spec-approved convention can collapse both scores to 0. The output must disclose this
choice so results are interpreted accordingly. Revising the approved metric convention
is outside this plan.

## Decision 6: Specify zero, tied-rank, and all-zero behavior

**Decision**: Use Wilcox zero handling (omit zero differences from signed ranks), retain
all images in the per-image table, assign average ranks to tied absolute differences,
and record zero and nonzero pair counts. If every difference is zero, report W=0,
p=1, W+=W-=0, and rank-biserial effect as null because its denominator is zero.

**Rationale**: This preserves the complete sample record while making the statistical
sample and undefined effect size explicit. It avoids presenting a no-difference case
as an estimated zero-sized effect.

## Decision 7: Keep diagnostics descriptive and deterministic

**Decision**: Report Shapiro-Wilk and Fisher-Pearson skewness over all paired
differences; generate a paired IoU plot and a difference-distribution plot. Diagnostics
do not replace the specified Wilcoxon test. State that median interpretation depends
on approximate symmetry. Record NumPy/SciPy/PyTorch versions and the Wilcoxon zero,
tie and method behavior.

**Rationale**: This matches the study's diagnostic guidance while keeping the primary
hypothesis and test fixed. With 800 observations, Shapiro-Wilk is within its supported
sample-size range. Constant differences produce unavailable diagnostics with a reason,
not a failure of the main analysis.

## Decision 8: Use existing dependencies and standard-library tests

**Decision**: Use SciPy already listed in `requirements.txt` for statistics and
`unittest` for focused tests. Add no package dependency. Test pure metric and pairing
helpers separately from the end-to-end checkpoint run.

**Rationale**: The repository already has the required numerical, statistics, and
plotting packages, and no existing automated test suite was found. Standard-library
tests avoid introducing a test dependency.

## Unresolved Questions

None. Defaults and edge-case conventions are defined by the approved specification and
repository behavior.
