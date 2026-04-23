# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-04-22 — Pipeline-audited release

Major rewrite after a full multi-stage academic-pipeline audit
(self-assessment + 5-reviewer review). This release supersedes the
original `bare_jrnl__3_.tex` draft.

### Added
- Multi-seed simulation driver (`code/simulation_driver.py`) that
  reproduces every number in the manuscript with a single command.
- Explicit **Simulation Setup and Validity Defence** section
  (Methods §3.1).
- Explicit **Data Provenance and Ethics** section (Methods §3.2).
- Explicit **Feature and Hyperparameter Selection** subsection with a
  per-parameter justification table (Methods §3.5).
- Cross-seed bootstrap 95 % confidence intervals on every reported
  metric.
- Pre-registered three-site pilot protocol (Future Work) with
  falsifiable hypotheses and a negative-result commitment.
- **Second author** (placeholder) and CRediT contribution statement.
- `README.md`, `AUTHORS.md`, `LICENSE`, `CHANGELOG.md`,
  `tests/test_driver_reproducibility.py`, and assorted docs.
- Self-assessment (`docs/01_SELF_ASSESSMENT.md`), reviewer reports
  (`docs/02_REVIEWER_REPORTS.md`), response-to-reviewers letter
  (`docs/03_RESPONSE_TO_REVIEWERS.md`).

### Changed
- Title reframed from clinical claim to in-silico feasibility study.
- Abstract rewritten to disclose the simulation-only nature of the
  evidence.
- "CNP scores" renamed to **Sectoral Retinal Abnormality Scores
  (SRAS)** because colour fundus imaging cannot directly measure CNP;
  the 48-sector vector is a surrogate after Joshi 2017.
- Monte Carlo Dropout framing separated from Bayesian hyperparameter
  optimisation; MCD is now correctly described as a posterior
  approximation, not an optimisation procedure.
- Power claim clarified: ~ 0.93 W **incremental** inference power
  above Pi 4 idle (total device power ~ 3.6 W). The original
  "0.6 W total" claim was incorrect.
- Synthetic-augmentation framing updated: DCGAN replaced by a
  diffusion-model target (FID < 10) in the deployed pipeline; the
  simulation driver uses a leakage-safe KDE surrogate.
- Comparison table deduplicated (Wilson 2023 appeared twice).
- Figure paths corrected (`fig:synthetic_real` and
  `fig:energy_efficiency` no longer point to the workflow diagram).

### Removed
- ≥ 12 unverifiable or mis-cited references (see
  `docs/04_REFERENCE_AUDIT.md` for the per-citation audit).
- Fabricated p-values, odds ratios, power-analysis numbers, and
  clinician-similarity t-tests that could not be derived from code.
- Inflated single-seed claims (original AUC 0.93 sensitivity 88 %
  were accidental; multi-seed mean is AUC 0.80, sensitivity 0.82).
- Deployment-economics and nurse-training sections moved to a short
  Planned Work subsection with future-tense verbs.

### Fixed
- Stratified train / val / test split to guarantee deaths in every
  test fold (prevents degenerate metrics).
- Augmentation applied only to the training fold (prevents leakage).
- Duplicate comparison-table row.
- Unused `\label` declarations.

## [1.0.0] — initial draft

Original single-author draft (`bare_jrnl__3_.tex`,
`references__13_.bib`) with AUC 0.93 / 88 % sensitivity claims. Found
to contain reference-integrity violations and simulation-disclosure
issues during the academic-pipeline audit. Superseded by 2.0.0.
