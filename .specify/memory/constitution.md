<!--
Sync Impact Report
Version change: uninitialized scaffold -> 1.0.0 (initial project constitution)
Modified principles: none
Added principles: Reproducibility and Traceability; Data Integrity and Leakage Prevention;
	Controlled Comparisons; Metrics and Statistical Reporting; Evidence-Led Conclusions
Added sections: Experimental Protocol; Development and Review Workflow
Removed sections: none
Follow-up TODO: confirm the original constitution ratification date
-->

# Salt Segmentation Experiment Constitution

## Core Principles

### I. Reproducibility and Traceability
Every reported experiment MUST record its code and configuration, dataset identity and
split, random seed, scenario, hyperparameters, environment, and resulting artifacts.
Runs MUST preserve logs and the applicable checkpoints, training history, and final
metrics. These records make results auditable and reproducible.

### II. Data Integrity and Leakage Prevention
Training, validation, and final test data MUST have explicit, documented roles. The
canonical external real test set MUST remain fixed and MUST NOT be used for training,
model selection, or synthetic-data generation. Synthetic samples MUST NOT derive from
test samples. Preprocessing MUST preserve label semantics; binary masks MUST use
nearest-neighbor interpolation.

### III. Controlled Comparisons
When comparing scenarios A and B, architecture, preprocessing, optimization settings,
data splits, and evaluation protocol MUST remain the same unless a documented protocol
explicitly declares otherwise. The training-data composition is the independent
variable for the A/B comparison. Any other difference MUST be recorded, and the
constitution does not presume that scenario B will outperform scenario A.

### IV. Metrics and Statistical Reporting
Primary segmentation results MUST include IoU and Dice on the fixed real test set.
Claims based on repeated runs MUST report the individual seeds and an appropriate
summary across those runs. Results from different test sets or evaluation protocols
MUST be labeled separately and MUST NOT be presented as directly comparable.

### V. Evidence-Led Conclusions
Conclusions MUST reflect the observed results and their limitations. Reports MUST
include material negative, null, and mixed findings alongside positive findings, and
MUST distinguish exploratory results from results intended to support the manuscript's
claims. This prevents selective reporting from overstating the evidence.

## Experimental Protocol

Each experiment MUST state its hypothesis, scenario, data sources, split, evaluation
set, seed plan, and relevant training settings before results are interpreted. The
canonical external test set MUST be used for comparisons intended to support the
project's downstream segmentation claims. A change to this protocol MUST be documented
with its rationale, and results from the changed protocol MUST be identified as such.

For binary segmentation, masks MUST remain discrete through loading and preprocessing.
The model input channel count and tensor dimensions MUST match the dataset and model
configuration. Dataset filters, exclusions, and synthetic-data generation procedures
MUST be recorded sufficiently to identify the resulting training pool.

## Development and Review Workflow

Changes to experiment code MUST preserve the protocol or explicitly document a protocol
change. Affected behavior MUST be checked with an appropriate test, smoke run, or other
focused validation before results are used. Training and evaluation outputs MUST be
associated with an identifiable run and retained in the project's results structure.

Before results are used in a report or manuscript, the analysis MUST verify scenario,
seed, dataset, test-set identity, and metric definitions. Updates to manuscript claims
and reviewer responses MUST be based on the verified results; unresolved experiments
MUST NOT be described as completed.

## Governance

This constitution governs experiment design, implementation, analysis, and reporting.
An amendment MUST state its rationale and impact, be reviewed by the project
maintainer(s), and update the version and amendment date. Completed experiments remain
interpretable under the protocol in effect when they were run; retrospective
re-evaluation MUST identify any changed protocol.

Constitution versions use MAJOR.MINOR.PATCH semantics: MAJOR changes remove or redefine
a principle or make an incompatible governance change; MINOR changes add a principle
or section, or materially expand requirements; PATCH changes clarify wording without
changing requirements. Compliance MUST be reviewed when an experiment is planned, when
its implementation or analysis changes the protocol, and before its results are
reported. Deviations MUST be documented with their rationale and impact.

**Version**: 1.0.0 | **Ratified**: TODO(RATIFICATION_DATE): original adoption date unavailable | **Last Amended**: 2026-11-06
