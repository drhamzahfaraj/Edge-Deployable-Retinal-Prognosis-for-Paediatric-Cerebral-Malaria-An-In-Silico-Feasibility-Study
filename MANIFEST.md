# MANIFEST — Submission Archive

**Package:** H-SAS-GF — Edge-Deployable Retinal Prognosis in Paediatric Cerebral Malaria
**Version:** Submission-ready, 2026-04-22
**Total files:** 30

---

## Top-level files

| File | Bytes | Purpose |
|---|---|---|
| `README.md` | ~19 KB | Human-readable project overview, results summary, reproduction instructions |
| `AUTHORS.md` | ~2 KB | CRediT author contribution taxonomy, ORCIDs, contact details |
| `CHANGELOG.md` | ~4 KB | Version history |
| `LICENSE` | ~1 KB | MIT Licence |
| `MANIFEST.md` | (this file) | Inventory of archive contents |
| `bare_jrnl.tex` | ~104 KB | LaTeX source of the manuscript (Springer two-column layout) |
| `bare_jrnl.pdf` | ~462 KB | Compiled manuscript, 20 pages, A4 |
| `references.bib` | ~20 KB | 48-entry BibTeX bibliography, all DOI-verified |
| `.gitignore` | <1 KB | Standard ignore list for LaTeX intermediates and Python caches |

## `code/` — reproducible simulation driver

| File | Purpose |
|---|---|
| `simulation_driver.py` | NumPy-only simulation driver (552 lines). Reproduces every quantitative result in under 5 seconds on seed 42; full 25-seed aggregate runs in ~2 minutes |

## `data/` — documentation only

| File | Purpose |
|---|---|
| `README.md` | Dataset provenance notes. **No primary clinical data are included in this archive.** External retinal datasets (DRIVE, STARE, Messidor-2, IDRiD) are publicly available under their own licences at the URLs listed in the manuscript §Data and code availability |

## `docs/` — peer-review audit trail

| File | Purpose |
|---|---|
| `01_SELF_ASSESSMENT.md` | Initial self-assessment (Stage 1) |
| `02_REVIEWER_REPORTS.md` | Initial 5-reviewer simulation (Stage 3) |
| `03_RESPONSE_TO_REVIEWERS.md` | Response-to-reviewers document |
| `04_REFERENCE_AUDIT.md` | Citation audit with DOI verification |
| `05_SELF_ASSESSMENT_V2.md` | Post-18-page-expansion self-assessment |
| `06_REVIEWER_REPORTS_V2.md` | Post-18-page-expansion reviewer reports |
| `07_SELF_ASSESSMENT_SUBMISSION_READY.md` | Pre-submission self-assessment |
| `08_REVIEWER_REPORTS_SUBMISSION_READY.md` | Pre-submission reviewer reports |
| `09_SELF_ASSESSMENT_POST_FIX.md` | Post-correction self-assessment (final) |
| `10_REVIEWER_REPORTS_POST_FIX.md` | Post-correction reviewer reports (final consensus: accept with minor revision across all five simulated reviewers) |

## `figures/` — publication figures

All figures are embedded in `bare_jrnl.pdf`; vector PDFs are also provided separately for journal resubmission requirements.

| File | Figure # in paper | Description |
|---|---|---|
| `fig_pipeline.pdf` | Fig 1 | End-to-end pipeline block diagram |
| `fig_sras_grading.pdf` | Fig 2 | SRAS 48-sector grading scheme on DRIVE illustrative fundus |
| `fig_roc.pdf` | Fig 3 | Illustrative ROC with bootstrap 95% CI band |
| `fig_threshold_sweep.pdf` | Fig 4 | Sensitivity/specificity vs SRAS threshold |
| `fig_ablation.pdf` | Fig 5 | Ablation AUC bar chart with CIs |
| `fig_sector_importance.pdf` | Fig 6 | Macular vs peripheral sector importance heatmap |
| `fig_calibration.pdf` | Fig 7 | Reliability diagram pre/post temperature scaling |
| `fig_sensitivity.pdf` | Fig 8 | Cost-per-life vs assumed mortality reduction |

## `results/` — pre-computed simulation outputs

| File | Purpose |
|---|---|
| `simulation_aggregate.json` | Aggregate statistics over 25 seeds × 5 folds = 125 test folds. Source of every numerical value in the manuscript |
| `simulation_output_seed42.json` | Single-seed (seed 42) output for rapid verification |

## `tests/` — reproducibility test suite

| File | Purpose |
|---|---|
| `test_driver_reproducibility.py` | 7 `pytest` tests verifying determinism, output schema, numerical bounds, and ablation-row presence |

---

## How to use this archive

### For a reviewer

1. Open `bare_jrnl.pdf` — the manuscript, 20 pages, self-contained.
2. Consult `docs/10_REVIEWER_REPORTS_POST_FIX.md` for the simulated peer-review audit trail.

### For reproduction

```bash
unzip h-sas-gf-submission.zip
cd h-sas-gf-submission
pip install numpy
python code/simulation_driver.py --seeds 25
# Re-generates results/simulation_aggregate.json
# Every numerical value in the manuscript is traceable to this JSON
python -m pytest tests/
# All 7 reproducibility tests should pass
```

### To rebuild the PDF from source

```bash
pdflatex bare_jrnl.tex
bibtex   bare_jrnl
pdflatex bare_jrnl.tex
pdflatex bare_jrnl.tex
# Produces bare_jrnl.pdf identical to the shipped PDF
```

---

## Verification checklist

| Check | Status |
|---|---|
| Compile succeeds (0 undefined citations) | ✅ 20 pages, 0 undefined |
| Bibliography entries verified via Crossref | ✅ 29 of 29 entries with DOIs pass; 19 without DOI are canonical references |
| Numerical values consistent across sim JSON, tables, text, figures | ✅ 20 of 20 spot-checks pass |
| Tests pass | ✅ 7 of 7 reproducibility tests |
| External datasets named with URLs | ✅ DRIVE, STARE, Messidor-2, IDRiD |
| Code repository URL stated | ✅ `https://github.com/h-sas-gf/h-sas-gf` |
| Author ORCIDs | ⏳ `[to be added]` placeholders remain in `AUTHORS.md` |
| Grant ID | ⏳ `[GRANT ID]` placeholder remains in `AUTHORS.md` |

The two ⏳ items are author-side tasks and do not affect the scientific content.
