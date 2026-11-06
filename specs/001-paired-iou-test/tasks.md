---

description: "Implementation tasks for paired per-image IoU analysis"
---

# Tasks: Paired Per-Image IoU Test

**Input**: Design documents from `specs/001-paired-iou-test/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`

**Tests**: Included because the implementation plan and quickstart specify standard-library `unittest` validation.

**Organization**: Tasks are grouped by user story. Implementation is a single CLI script; tests use one focused module.

## Phase 1: Setup

**Purpose**: Establish the feature's test entry point without adding dependencies.

- [ ] T001 Create `Salt-Segmentation-UNet/tests/test_paired_iou_test.py` with a discoverable `unittest.TestCase` and import smoke-test structure.

---

## Phase 2: Foundational

**Purpose**: Create the shared CLI module and model-loading boundary required by all user stories.

- [ ] T002 Create `Salt-Segmentation-UNet/paired_iou_test.py` with an executable `main()` entry point, imports from `utils.config` and `utils.model.UNet`, and a helper that loads a state-dict checkpoint with `map_location` and rejects incompatible weights.

**Checkpoint**: The CLI module imports successfully and both supplied checkpoint files can be validated against the existing `UNet` architecture without starting training.

---

## Phase 3: User Story 1 - Executar comparação pareada (Priority: P1) 🎯 MVP

**Goal**: Evaluate both pretrained models on exactly the same real test image/mask IDs and produce per-image metrics without training.

**Independent Test**: Using temporary images/masks and compatible test model outputs, confirm stable ID matching, identical sample membership for both models, and one result row per valid ID; malformed or incomplete pairs fail without a success output.

### Tests for User Story 1

- [ ] T003 [US1] Add tests in `Salt-Segmentation-UNet/tests/test_paired_iou_test.py` for stable ID pairing, duplicate/missing image-mask IDs, canonical test-directory defaults, grayscale 128x128 preprocessing, nearest-neighbor binary masks, per-image IoU/Dice, and the spec-approved IoU=0 empty-union convention.

### Implementation for User Story 1

- [ ] T004 [US1] Implement canonical local/HPC test-directory defaults, image/mask discovery keyed by unique file stem, validation that all pairs are complete, and deterministic ID sorting in `Salt-Segmentation-UNet/paired_iou_test.py`; do not use `Salt-Segmentation-UNet/output/test_paths.txt` as the canonical source.
- [ ] T005 [US1] Implement batched inference of both loaded checkpoints on every validated test ID, compute per-image IoU and Dice using the shared ground-truth mask, and write stable-ID-sorted `per_image_metrics.csv` in `Salt-Segmentation-UNet/paired_iou_test.py`.

**Checkpoint**: The MVP evaluates the two default checkpoints without training and writes exactly one paired metric row per test ID; invalid pairs or checkpoints stop with an actionable error.

---

## Phase 4: User Story 2 - Interpretar evidência estatística (Priority: P1)

**Goal**: Report the prespecified Wilcoxon result, effect size, descriptive statistics, diagnostics, and plots from per-image IoU differences.

**Independent Test**: Feed known paired metric vectors to the statistical helpers and verify the alternative direction, p-value decision, rank sums, rank-biserial effect, zero/tie behavior, and all-zero behavior.

### Tests for User Story 2

- [ ] T006 [US2] Add statistical tests in `Salt-Segmentation-UNet/tests/test_paired_iou_test.py` for one-sided `greater` Wilcoxon, average ranks for tied absolute differences, excluded zero differences, all-zero inputs (`W=0`, `p=1`, effect `null`), and unavailable Shapiro/skewness diagnostics for constant inputs.

### Implementation for User Story 2

- [ ] T007 [US2] Implement per-model medians/IQRs, median IoU difference, favorable-pair percentage, Wilcoxon signed-rank statistics, positive/negative rank sums, zero/nonzero counts, and rank-biserial effect in `Salt-Segmentation-UNet/paired_iou_test.py`; preserve all rows and return a null effect when all differences are zero.
- [ ] T008 [US2] Add Shapiro-Wilk and Fisher-Pearson skewness diagnostics plus `paired_iou.png` and `iou_difference.png` generation in `Salt-Segmentation-UNet/paired_iou_test.py`; write `summary.json` with the effective settings, input paths, versions, tested IDs, and empty-union convention.

**Checkpoint**: Given valid per-image pairs, the CLI emits all required descriptive and inferential results without switching away from Wilcoxon based on normality diagnostics.

---

## Phase 5: User Story 3 - Ajustar e reproduzir a execução pela linha de comando (Priority: P1)

**Goal**: Run the same analysis with defaults or override inputs and statistical/inference parameters without editing source code.

**Independent Test**: Verify defaults point to the specified checkpoints and canonical test set; override checkpoint, test/output directories, threshold, alpha, alternative, batch size, and device; reject invalid values before inference.

### Tests for User Story 3

- [ ] T009 [US3] Add CLI tests in `Salt-Segmentation-UNet/tests/test_paired_iou_test.py` for canonical Windows/HPC test-directory defaults, valid path overrides, invalid threshold/alpha/alternative/batch/device values, and missing checkpoint/test paths.

### Implementation for User Story 3

- [ ] T010 [US3] Implement and integrate `argparse` options in `Salt-Segmentation-UNet/paired_iou_test.py` for `--base-checkpoint`, `--aug-checkpoint`, `--test-dir`, `--output-dir`, `--threshold`, `--alpha`, `--alternative`, `--batch-size`, and `--device`; set canonical local/HPC TESTE A test-directory defaults, validate values before inference, and record effective values in `summary.json`.

**Checkpoint**: The default command reproduces TESTE A, and valid CLI overrides take effect without source edits while invalid arguments exit before reporting results.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Keep the validation guide aligned with the final CLI and verify the complete flow.

- [ ] T011 Update `specs/001-paired-iou-test/quickstart.md` to match the final CLI help, output fields, checkpoint defaults, and canonical-test invocation in `Salt-Segmentation-UNet/paired_iou_test.py`.
- [ ] T012 Run `Push-Location Salt-Segmentation-UNet; python -m unittest discover -s tests -p "test_paired_iou_test.py" -v; Pop-Location` and execute the end-to-end command from `specs/001-paired-iou-test/quickstart.md`; verify 800 paired rows, exact ID equality with `D:\dataset\tgs-salt\subset_split\test_set.csv`, and all documented output files.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No prerequisites; creates the focused test module.
- **Foundational (Phase 2)**: Depends on Setup; creates the importable CLI and checkpoint loader.
- **User Story 1 (Phase 3)**: Depends on Foundation; establishes validated image pairing and per-image outputs.
- **User Story 2 (Phase 4)**: Depends on User Story 1 because the statistical analysis consumes its per-image IoU rows.
- **User Story 3 (Phase 5)**: Depends on the analysis from User Stories 1 and 2; exposes configurable inputs and parameters for the complete run.
- **Polish (Phase 6)**: Depends on all three user stories.

### User Story Dependencies

- **US1 (P1)**: Can start after Foundation; no dependency on another user story. This is the MVP.
- **US2 (P1)**: Requires US1's per-image IoU differences.
- **US3 (P1)**: Integrates paths and settings with the completed US1/US2 analysis.

### Parallel Opportunities

There are no safe parallel implementation tasks within a user story: each story changes the same CLI module and shared test module, and tests precede implementation. Keep the story phases sequential to avoid conflicting edits. Independent review of the quickstart may occur while the final end-to-end run is prepared, but no task IDs are marked `[P]` because there is no independent file-writing task in this plan.

### Parallel Execution Examples per User Story

- **US1**: No parallel task pair; run T003 before T004-T005 to preserve the test-first sequence.
- **US2**: No parallel task pair; run T006 before T007-T008 because all tests and statistical helpers share the same modules.
- **US3**: No parallel task pair; run T009 before T010 to establish CLI behavior before implementation.

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup and Foundation.
2. Complete US1 tests and implementation.
3. Validate paired inference and the 800-row CSV independently; do not claim statistical significance until US2 is implemented.

### Incremental Delivery

1. Deliver US1 as the paired per-image inference MVP.
2. Add US2 statistical interpretation and diagnostic outputs.
3. Add US3 configurable CLI parameters and defaults for reproducible reruns.
4. Run the full unit suite and quickstart end-to-end check in Polish.

## Notes

- Every task has a sequential ID, checkbox, exact file path, and story label where required.
- No task creates a web interface, starts model training, adds a dependency, or changes the existing `evaluate.py` behavior.
- The end-to-end validation requires access to the canonical 800-image test data and both checkpoint files.
