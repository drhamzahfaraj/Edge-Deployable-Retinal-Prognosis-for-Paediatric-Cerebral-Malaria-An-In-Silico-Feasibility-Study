# H-SAS-GF

**Edge-Deployable Retinal Prognosis for Paediatric Cerebral Malaria:
An In-Silico Feasibility Study**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Reproducible](https://img.shields.io/badge/reproducible-seed%3D42-green.svg)]()
[![Status](https://img.shields.io/badge/status-simulation--only-orange.svg)]()

> **One-sentence summary:** an open, reproducible in-silico feasibility
> study of a hybrid SSL + GAT + XGBoost pipeline for paediatric
> cerebral-malaria retinal prognosis on a \$50 single-board computer.
> All numbers in the paper are re-derivable with a single command.

> **No**  patient was enrolled, no Raspberry Pi was physically measured, 
>no clinician scored an image. The repository is a pre-registration of a pilot
> whose results will be reported here regardless of outcome.

---

## Table of contents

1. [Authors and contact](#1-authors-and-contact)
2. [Abstract](#2-abstract)
3. [Key results and findings](#3-key-results-and-findings)
4. [Ablation and benchmarking](#4-ablation-and-benchmarking)
5. [Contribution](#5-contribution)
6. [Methodology (and mechanism of compute / energy saving)](#6-methodology)
7. [Datasets](#7-datasets)
8. [Repository structure](#8-repository-structure)
9. [Running the experiments](#9-running-the-experiments)
10. [Limitations](#10-limitations)
11. [Citation](#11-citation)
12. [License](#12-license)
13. [Acknowledgments](#13-acknowledgments)

---

## 1. Authors and contact

| Role | Author | Affiliation | Contact |
|---|---|---|---|
| Corresponding author | **Hamzah Faraj** | Department of Science and Technology, Ranyah College, Taif University, Taif 21944, Saudi Arabia | <f.hamzah@t.edu.sa> |
| Co-author | **Yassine Aribi** | Research Groups in Intelligent Machines (REGIM Laboratory), National Engineering School of Sfax (ENIS), University of Sfax, Tunisia | <yassine.aribi@ieee.org> |

**Author contributions (CRediT taxonomy).**
H.F.: conceptualisation, methodology, software, formal analysis,
investigation, writing (original draft + review), visualisation,
project administration.
Y.A.: clinical validation, data curation, resources,
supervision, funding acquisition, writing (review & editing).
Both authors read and approved the final manuscript.

---

## 2. Abstract

Cerebral malaria caused an estimated 619 000 deaths in 2021,
disproportionately among children under five in sub-Saharan Africa.
Retinal imaging carries diagnostic and prognostic information for CM,
but rural clinics lack both imaging devices and specialist graders.
We present **H-SAS-GF**, a hybrid pipeline combining self-supervised
representation learning (SSL), a Graph Attention Network (GAT) over
48 retinal sectors, and a gradient-boosted classifier (XGBoost),
designed to run on a \$50 single-board computer. This work is an
**in-silico feasibility study**: every numerical result is produced by
a reproducible simulation whose priors are anchored to published
paediatric CM cohorts. Aggregated over 25 random seeds, the pipeline
produces AUC 0.80 (95 % CI 0.51–0.95) with a mean sensitivity of 0.82
at threshold 0.6. A modelled compute budget for a quantised TFLite
deployment yields ≈ 4.5 s inference at ≈ 0.93 W incremental power and
a 600 KB model. No patient was enrolled and no device was physically
measured; clinical validation is pre-registered as future work.

---

## 3. Key results and findings

All numbers below come from 25-seed multi-seed aggregation produced by
`code/simulation_driver.py` with `--seed 42 --n-seeds 25`. Single-seed
runs can and do produce different numbers — this is the honest story
of small-n evaluation.

| Metric | Mean | SD | 95 % CI |
|---|---|---|---|
| AUC | **0.80** | 0.12 | [0.51, 0.95] |
| Sensitivity @ 0.6 | **0.82** | 0.28 | [0.25, 1.00] |
| Specificity @ 0.6 | 0.54 | 0.27 | [0.06, 1.00] |
| F1 @ 0.6 | 0.43 | 0.12 | [0.24, 0.66] |
| Macular sector importance | 0.72 | 0.08 | [0.59, 0.86] |
| Peripheral sector importance | 0.28 | 0.08 | [0.14, 0.41] |

Headline takeaway: the macular / peripheral importance ratio is stable
across seeds, but the AUC CI is wide because the test fold contains
only ~ 4 deaths and a single accidental false negative shifts
sensitivity by 25 percentage points. The multi-seed mean is the
defensible number; a single-seed AUC of 0.93 like in some prior draft
is an accident, not a result.

**Modelled compute budget** (datasheet-derived, not physically measured):

| Device | Inference time | Incremental power | Total device power | Model size |
|---|---|---|---|---|
| Raspberry Pi 4 (1 GB) | ≈ 4.5 s | ≈ 0.93 W | ≈ 3.6 W | 600 KB |
| Raspberry Pi Zero 2 W | ≈ 13 s | ≈ 0.7 W | ≈ 1.2 W | 600 KB |
| Jetson Nano | ≈ 0.8 s | ≈ 5 W | ≈ 10 W | 600 KB |

**Cost model** (50 000 tests over 5 years, Malawi parameters):

| Item | Value |
|---|---|
| Setup cost | \$5 050 |
| Per-test cost | \$0.50 |
| Total cost | \$30 050 |
| Modelled lives saved | ≈ 220 |
| Modelled cost per life saved | ≈ \$135 |

---

## 4. Ablation and benchmarking

**Ablation** (25 seeds; ΔAUC is relative to the full model):

| Configuration | AUC (mean ± SD) | 95 % CI | ΔAUC |
|---|---|---|---|
| **Full H-SAS-GF (SSL + GAT + XGBoost)** | **0.80 ± 0.12** | [0.51, 0.95] | — |
| — without SSL | 0.73 ± 0.13 | [0.53, 0.96] | −0.07 |
| — without GAT (uniform sector weights) | 0.72 ± 0.17 | [0.35, 0.93] | −0.08 |
| — without XGBoost (plain logistic head) | 0.70 ± 0.14 | [0.46, 0.90] | −0.10 |

Each block contributes. The XGBoost boosting head has the largest
single-block effect on this prior structure. No per-ablation p-values
are reported; the honest statistic for paired comparisons on the same
test fold is bootstrap ΔAUC with CI, which is what the table shows.

**Benchmarking against published paediatric CM retinal-AI baselines:**

| Method | AUC | Sn | Sp | Power (W) | Device class |
|---|---|---|---|---|---|
| Manual MR grading [Beare 2006] | N/A | ~0.70 | ~0.65 | 0 | Clinician |
| Beare 2009 FA-based CNP | N/A | ~0.68 | ~0.63 | — | FA imaging |
| Joshi 2017 CNN for MR detection | ~0.90 | — | — | ~10 | Workstation |
| Kurup 2023 transfer learning (MR) | ~0.93 | — | — | ~10 | Workstation |
| Rajaraman 2019 CNN (parasite detection) | 0.95 | 0.92 | 0.92 | ~10 | Workstation |
| **H-SAS-GF (simulated, this work)** | **0.80** | **0.82** | **0.54** | **~0.93¹** | **Pi 4** |

¹ Incremental inference power above Pi 4 idle; total device power ≈ 3.6 W.

H-SAS-GF trades several AUC points for a ~ 10× reduction in compute
envelope. It is not the most accurate system on this list; it is
plausibly the only one that can run on a \$50 device with solar
backup in a clinic that does not have reliable mains power.

---

## 5. Contribution

Three contributions, in order of importance:

1. **An open, reproducible simulation pipeline** whose priors are
   anchored to published paediatric CM cohorts. Every number in the
   paper can be re-derived in under five seconds on a laptop.
2. **A hybrid SSL + GAT + XGBoost architecture** justified against
   the ≤ 1 GB RAM, ≤ 1 W incremental compute budget of a quantised
   TFLite deployment. Feature / hyperparameter choices are explicit
   (see Methods §3.3 in the paper).
3. **A pre-registered three-site pilot protocol** (Blantyre, Kisumu,
   Ibadan) specifying the falsifiable hypotheses that the simulation
   generates for real-world data, with a negative-result commitment
   to publish whatever the pilot produces.

The paper does **not** claim a clinical result. The simulation is
deliberately designed so that what it does establish (feasibility,
ablation, design sensitivity) is separated from what it does not
(clinical utility).

---

## 6. Methodology

### 6.1 Simulation setup and validity defence

Every numerical result in the paper is produced by
`code/simulation_driver.py`. Its inputs are:

| Input | Source |
|---|---|
| Cohort priors | Beare 2006 AJTMH; Beare 2009 JID; Joshi 2017 Sci Rep; Barrera 2018 eLife; MacCormick 2020 JID |
| Random seed | base seed 42, aggregated over 25 seeds |
| Compute cost model | Raspberry Pi 4 datasheet + FLOP count of quantised TFLite model |
| Cost model | WHO published Malawi per-day hospital costs |

No patient was enrolled. No Raspberry Pi was measured. No clinician
scored an image. The simulation is defended on four pillars:
(1) honest framing as feasibility, not clinical utility;
(2) reproducibility via fixed seeds and released code;
(3) prior anchoring to DOI-verified published cohorts;
(4) explicit limitations including a retraction criterion.

### 6.2 Hybrid pipeline

```
          ┌──────────────────────────────────────────────────┐
          │  Fundus image  64×64, normalised to [0,1]        │
          └────────────────────┬─────────────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │  Block 1 — SSL feature extractor            │
        │  EfficientNet-Lite backbone                 │
        │  SimCLR contrastive pretraining (τ = 0.5)   │
        │  Output: 128-dim embedding                  │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │  Block 2 — Graph Attention Network          │
        │  48-node graph (macula 1-24, periphery      │
        │  25-48), 2 attention heads                  │
        │  Output: weighted 48-dim sector score       │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │  Block 3 — Gradient-boosted classifier      │
        │  XGBoost, 30 trees, max depth 3             │
        │  Output: probability of death, in [0,1]     │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │  Monte Carlo Dropout ×10 → variance U       │
        │  flag if U > 0.02 for clinician review      │
        └─────────────────────────────────────────────┘
```

### 6.3 Mechanism of compute / energy saving

Three layered reductions yield the ≈ 0.93 W incremental power and
600 KB model size:

| Layer | Reduction | How |
|---|---|---|
| Input | 20× | 256 × 256 RGB → 64 × 64 resize, chosen from a sweep where AUC plateaus above 64 × 64 |
| Feature | 3× | MobileNetV2/EfficientNet-Lite instead of full EfficientNet-B0 or Vision Transformer |
| Weights | 4× | int8 post-training quantisation via TFLite |
| Classifier | 10–50× | 30 shallow XGBoost trees instead of a full transformer head |

The result is a 1.5 GFLOP inference path that fits in a 600 KB
quantised model. On a Pi 4 sustaining ~ 2.0 GFLOP/s in int8, this is
~ 0.75 s of compute, with image I/O and pre-processing bringing the
wall-clock to ~ 4.5 s end-to-end.

### 6.4 Feature and hyperparameter selection (summary)

Every hyperparameter is justified against one of three criteria:
clinical convention, hardware budget, or an empirical sweep. See
Table 2 in the paper (`Hyperparameter → Justification`) for the
full table. A reviewer asking "why 48 sectors, why 128 dim, why 30
trees?" gets a line-by-line answer.

### 6.5 Synthetic augmentation

500 synthetic samples are drawn from the **training fold only** (no
leakage). In the deployed pipeline these would be from a denoising
diffusion probabilistic model targeting FID < 10; in the released
simulation driver we use a kernel-density estimator over the training
distribution as the leakage-safe surrogate, so reviewers can run the
driver without a trained generator.

---

## 7. Datasets

**Primary (simulated):** cohort priors from Beare 2006 AJTMH; Beare
2009 JID; Joshi 2017 Sci Rep; Barrera 2018 eLife; MacCormick 2020 JID.
No patient-level data are used, stored, or released.

**External robustness (publicly available, real datasets):**

| Dataset | Size | Role | URL |
|---|---|---|---|
| DRIVE | 40 images | Vessel segmentation, feature-extractor transfer test | <https://drive.grand-challenge.org> |
| STARE | 20 images | Second, independent vessel dataset | <https://cecas.clemson.edu/~ahoover/stare/> |
| Messidor-2 | 1748 images | Diabetic-retinopathy robustness check | <https://www.adcis.net/en/third-party/messidor2/> |
| IDRiD | 597 images | Indian DR dataset, distribution-shift probe | <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> |

These are **not** CM datasets. Their only role is to test whether
the SSL-pretrained feature extractor generalises across camera type
and pathology class. A real external CM validation is pre-registered
as future work.

---

## 8. Repository structure

```
h-sas-gf/
├── README.md                         ← this file
├── LICENSE                           ← MIT
├── AUTHORS.md                        ← author list + CRediT
├── CHANGELOG.md                      ← version history
├── bare_jrnl.tex                     ← the revised manuscript
├── references.bib                    ← DOI-verified bibliography
│
├── code/
│   └── simulation_driver.py          ← one-command reproducibility
│
├── results/
│   └── simulation_aggregate.json     ← output of the driver
│
├── data/
│   └── README.md                     ← no patient data; priors only
│
├── figures/
│   └── README.md                     ← figures regenerated from results
│
├── tests/
│   └── test_driver_reproducibility.py
│
└── docs/
    ├── 01_SELF_ASSESSMENT.md         ← Stage 1 critical analysis
    ├── 02_REVIEWER_REPORTS.md        ← Stage 3 five-reviewer reports
    ├── 03_RESPONSE_TO_REVIEWERS.md   ← revision response letter
    ├── 04_REFERENCE_AUDIT.md         ← detailed bibliography audit
    └── 05_PROCESS_RECORD.md          ← pipeline process summary
```

---

## 9. Running the experiments

**Prerequisites:** Python ≥ 3.9, numpy. Nothing else.

```bash
# Clone
git clone https://github.com/h-sas-gf/h-sas-gf.git
cd h-sas-gf

# (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install numpy

# Reproduce every number in the paper (25 seeds, ~ 4 s)
python code/simulation_driver.py --seed 42 --n-seeds 25 \
       --outdir results

# The aggregate output is written to
# results/simulation_aggregate.json
```

All numbers in Tables 1–5 of the paper are derivable from that JSON
file. To see them in a human-readable format:

```bash
python -c "
import json
r = json.load(open('results/simulation_aggregate.json'))
for section in ['main_metrics', 'ablation', 'compute_budget_modelled',
                'cost_model']:
    print(f'\n== {section.upper()} ==')
    for k, v in r[section].items():
        print(f'  {k}: {v}')
"
```

### Single-seed run

```bash
python code/simulation_driver.py --seed 42 --n-seeds 1
```

Useful for debugging but **not** what the paper reports. Single-seed
AUC in this setting ranges from 0.51 to 0.95 across seeds; the
multi-seed mean with CIs is the reported number.

### Sweep over seeds

```bash
for s in 0 42 100 1000 2024; do
  python code/simulation_driver.py --seed $s --n-seeds 25 \
         --outdir results/seed_$s
done
```

---

## 10. Limitations

1. **Simulation only.** No patient was enrolled; no Raspberry Pi was
   physically measured; no clinician scored an image.
2. **Prior dependence.** Cohort priors are drawn from a single
   paediatric Malawi cohort. Priors from Kenya, Uganda, Nigeria,
   Bangladesh may differ.
3. **Small n.** With n = 132 and 18 deaths, test-fold variance is
   wide. Multi-seed aggregation mitigates but does not eliminate this.
4. **SRAS is a surrogate.** True capillary non-perfusion requires
   fluorescein angiography (rarely available rurally). Our
   sectoral-retinal-abnormality score is a colour-fundus surrogate
   following Joshi 2017's automated whitening / hemorrhage pipeline.
5. **Modelled hardware.** The ~ 0.93 W incremental power and ~ 4.5 s
   inference are datasheet-derived, not measured.
6. **Retraction criterion.** If the pre-registered pilot (Blantyre,
   Kisumu, Ibadan) produces AUC below 0.70 with a 95 % CI excluding
   0.80, or if the macular-over-peripheral importance pattern
   inverts on real data, the architecture in its present form should
   be withdrawn. Negative results will be published regardless.

---

## 11. Citation

```bibtex
@article{FarajAribiHSASGF2026,
  author  = {Faraj, Hamzah and Aribi, Yassine},
  title   = {{H-SAS-GF}: An In-Silico Feasibility Framework for
             Scalable Cerebral Malaria Retinal Prognosis Using
             Small-Cohort Priors and Synthetic Augmentation},
  journal = {Medical & Biological Engineering & Computing (Springer)},
  year    = {2026},
  note    = {Simulation-based feasibility study; code at
             \url{https://github.com/h-sas-gf/h-sas-gf}}
}
```

---

## 12. License

This repository is released under the **MIT License**. See
[LICENSE](LICENSE) for the full text.

The external datasets (DRIVE, STARE, Messidor-2, IDRiD) are
distributed under their own licences; the repository does **not**
redistribute them. You must obtain them separately from the URLs in
§7.

---

## 13. Acknowledgments

The author would like to acknowledge the Deanship of
Graduate Studies and Scientific Research, Taif Univer-
sity for funding this work.
