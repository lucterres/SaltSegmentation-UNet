# Paired Per-Image IoU Test Implementation Plan

**Branch**: `001-paired-iou-test` | **Date**: 2026-11-06 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-paired-iou-test/spec.md`

## Summary

Add one command-line script alongside the existing experiment scripts. It loads the
real-only and real-plus-synthetic U-Net checkpoints, evaluates both on identical real
test images, calculates per-image IoU/Dice, and runs the specified paired Wilcoxon
analysis. It does not train models, add a web interface, or change `evaluate.py`.

## Technical Context

**Language/Version**: Python 3.8.16

**Primary Dependencies**: Existing PyTorch 2.4.1, NumPy, SciPy >=1.11, pandas,
OpenCV, and Matplotlib. Use the standard-library `unittest`; add no dependencies.

**Storage**: CSV/JSON/PNG files on the local or HPC filesystem.

**Testing**: `unittest` for metrics, pair validation, statistical edge cases, and CLI
arguments; one documented end-to-end CLI run against the canonical 800-image test set.

**Target Platform**: Existing Linux HPC environment; CPU and CUDA are selected or
overridden through the CLI.

**Project Type**: Standalone Python CLI script in the existing segmentation project.

**Performance Goals**: Process the canonical 800 images in configurable batches; no
hard runtime target is specified.

**Constraints**: Reuse `utils.config`, `utils.model.UNet`, and existing grayscale,
128x128, `[0,1]`, nearest-neighbor mask preprocessing conventions. Defaults are the
two specified checkpoints, canonical real test set, threshold 0.5, alpha 0.05, and
one-sided `greater`. Match pairs by unique image ID and reject incomplete pairs. Apply
the spec-approved IoU=0 convention when union is empty, although the TeX sample helper
uses IoU=1 for empty-empty. No web interface, training flow, or change to
run-level `evaluate.py`.

**Scale/Scope**: One checkpoint per scenario, one fixed set of 800 real test images,
per-image metrics and a single paired statistical analysis.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Reproducibility and Traceability**: PASS. Save effective CLI values, checkpoint
  paths, versions, metric convention, and evaluated IDs with the outputs.
- **Data Integrity and Leakage Prevention**: PASS. Test data is used only for
  evaluation; no training or synthetic generation is performed.
- **Controlled Comparisons**: PASS. Both models receive the same test IDs and
  preprocessing; only the pretrained checkpoint differs.
- **Metrics and Statistical Reporting**: PASS. Report per-image IoU and Dice; use IoU
  alone for the prespecified Wilcoxon test and report its effect size and diagnostics.
- **Evidence-Led Conclusions**: PASS. Report the test decision and limitations without
  presuming the augmented model is better.
- **Experimental Protocol**: PASS. Dataset, checkpoints, threshold, and test direction
  are fixed defaults and overridable CLI arguments are recorded.
- **Development and Review Workflow**: PASS. Focused unit tests and the quickstart
  validate outputs before they are reported.

No gate violations. The discrepancy between the spec-approved empty-union IoU=0 and
the TeX helper's empty-empty IoU=1 is recorded in `research.md`; this plan retains the
approved spec convention.

## Project Structure

### Documentation (this feature)

```text
specs/001-paired-iou-test/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md                 # Created by /speckit-tasks
```

### Source Code (repository root)

```text
Salt-Segmentation-UNet/
├── paired_iou_test.py       # One CLI, following existing argparse scripts
└── tests/
    └── test_paired_iou_test.py
```

**Structure Decision**: Keep the feature to one CLI script and one focused test module
under the existing Python project. Do not add a web interface, package layer, service,
or new runtime dependency. Document the CLI arguments in `quickstart.md` rather than
creating a separate API contract artifact.

## Complexity Tracking

No constitution violations. A single dedicated script avoids mixing image-level
pairing with the existing `evaluate.py` run-level, cross-seed summary and paired t-test.
