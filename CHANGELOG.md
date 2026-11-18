# Changelog

## v0.1.0 (2026-11-18)

First versioned source release of ReproPerturb.

- Stage-1 SoftVQ cell-state representation and Stage-2 prototype mass flow with signed pathway conditioning and matched add-one supervision.
- Validation-selected best checkpoints for the pathway, composition and expression workflows.
- Kernel relation scoring and direct ranking of 34 signed pathway candidates across three seeds.
- Direct composition prediction and per-seed evaluation for nine methods, including PerturbGen. Metrics include MAE, RMSE, Jensen–Shannon divergence, Hellinger distance and dominant cell-type accuracy.
- Task-specific expression training and evaluation for ReproPerturb, with checkpoint selection by validation gene-wise Pearson correlation.
- Versioned installation, asset requirements, training and evaluation instructions in the [README](README.md).

The release provides source and configuration files. Model checkpoints, data and prepared benchmark inputs are obtained separately.
