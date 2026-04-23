# Data directory

## Patient data

**This repository contains no patient-level data.**

The cerebral-malaria prognosis task in this study is evaluated on a
*simulated* cohort whose distributional priors are drawn from
published summary statistics (Beare 2006 AJTMH; Beare 2009 JID; Joshi
2017 Sci Rep; Barrera 2018 eLife; MacCormick 2020 JID). The priors
are hard-coded in the `Config` dataclass of
`../code/simulation_driver.py`; no re-identifiable patient data is
used, stored, or released.

## External robustness datasets

These are public retinal datasets referenced in the paper for
feature-extractor transfer tests. They are **not redistributed** here;
obtain each from its primary source, subject to its own licence.

| Dataset | Size | Use in this paper | Obtain from |
|---|---|---|---|
| DRIVE | 40 images | SSL pretraining / vessel transfer | <https://drive.grand-challenge.org> |
| STARE | 20 images | SSL pretraining / vessel transfer | <https://cecas.clemson.edu/~ahoover/stare/> |
| Messidor-2 | 1748 images | Downstream DR robustness probe | <https://www.adcis.net/en/third-party/messidor2/> |
| IDRiD | 597 images | Downstream DR robustness probe | <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> |

## Reproducible simulation output

`../results/simulation_aggregate.json` is produced by the driver and
contains every number reported in the paper. Run
`python ../code/simulation_driver.py --seed 42 --n-seeds 25` to
regenerate it (< 5 seconds on a laptop).

## For the pre-registered real-world pilot

When the Blantyre / Kisumu / Ibadan pilot begins (see §7 of the
paper), any real patient data will be stored on the partner-institution
servers and will not enter this repository. The repository will be
updated only with de-identified summary statistics after IRB approval.
