# Learned stabilization models and numerical evidence

[![Model verification](https://github.com/ehsanakrami/mltau-laminar-models-data/actions/workflows/verify.yml/badge.svg)](https://github.com/ehsanakrami/mltau-laminar-models-data/actions/workflows/verify.yml)
[MIT licence](LICENSE) · [Archived data DOI](https://doi.org/10.5281/zenodo.24046881)

Trained coefficient maps, saved numerical data and verification tools accompanying **Learned Stabilization of Incompressible Flow: Adaptivity, Reliability and Deployment Cost**, by Ehsan Akrami and Yan Delauré.

**Release:** version 0.1.1, archived at [doi:10.5281/zenodo.24046881](https://doi.org/10.5281/zenodo.24046881) under the MIT licence. The manuscript remains under author review. Publication of the artifacts does not establish general stabilization superiority.

The repository supports inspection of trained parameters, inference on computed predictor features, saved-data reanalysis and a finite-dimensional pressure certificate. The core FEM/fluid solver, solver binaries and training implementation are excluded. Running inference is not a new flow simulation.

## Start here

```bash
python -m pip install -e .
python examples/predict_pressure45.py
python -m unittest discover -s tests -v
# Optional independent finite-data certificate
python -m pip install -e '.[verification]'
python verification/linear_pressure/verify_pressure_certificate.py
```

To infer from a saved rectangular Q1 predictor field:

```bash
python -m pip install -e '.[fields]'
python examples/predict_saved_field.py tests/fixtures/predictor_q1_mms_n10_nu006.e --nu 0.006 --output coefficients.csv
```

The supplied field is an archived computed SH predictor. This example computes its 24 features and coefficients; it does not run the fluid solver.

Use Python 3.10 or later. NumPy is sufficient for model inference. SciPy is needed for the matrix certificate; netCDF4 is needed only for optional saved-field feature extraction. Tests perform no network access and require no private solver.

## Collections and scientific scope

| Collection | Contents | Interpretation |
|---|---|---|
| `models/pressure45` | Selected continuous-feature ridge and seven matched model/control candidates | 45 qualified training teachers; validation-selected ridge. Fresh parameters in exposed families, not unrestricted flow-family validation. |
| `data/pressure45` | Saved qualified feature/teacher arrays and complete planned-case catalogue | Planned failed cases remain in the catalogue; arrays are supplied only where qualified. |
| `experimental/shared58` | Earlier global response coefficients and complete method/profile result tables | Restricted named pressure gains; fixed geometry, advection and mesh. |
| `experimental/shared62` | Expanded coefficient candidates, thirteen-setting physical readbacks and metrics | Registered partial checks pass, but added complete pressure/velocity mixture tests fail at all thirteen fitting settings. |
| `verification/linear_pressure` | Stored affine matrices and standalone roundoff-bounded checker | Fixed binary64 linear-response obstruction; not a continuum theorem or exclusion of nonlinear closures. |
| `analysis/rectangular_features` | Source-free saved-field postprocessing routines | Static axis-aligned Q1 rectangles only; this is not the private solver. |

The selected ridge wins 27/32 joint Shakib comparisons; its matched constant wins 28/32. Conditional mean pressure gains do not establish greater all-case reliability. Equal-cost superiority is not established. Experimental file labels containing `NOT_RELEASED` are retained archival identifiers; they do not indicate scientifically qualified production models. See [model card](docs/MODEL_CARD.md) and [coverage](docs/REPRODUCIBILITY.md).

## Correct inference

The map takes 24 physical features from a converged computed SH predictor, applies the stored asinh normalization and clipping, and returns a coefficient bounded between 0.004 and 30. The output is a coefficient multiplier in the paper's declared stationary solver contract; it is not a dimensional tau value or a velocity/pressure solution. Do not substitute arbitrary 24-column data or exact-solution features. A matched constant does not need a predictor. Feature order is defined in `docs/feature_schema.json`; executable formulas are in `analysis/rectangular_features`. The [integration contract](docs/INTEGRATION.md) explains how the coefficient multiplies both stabilization channels and remains frozen during the corrected solve.

## Data and versioning

Small trained coefficients, qualified teacher arrays, all available predictor features, inference fixtures and key result tables are in Git. Three screened numerical archives (435,798,986 bytes) are deposited on Zenodo with checksum readback. Their inventories and public download URLs are in `data/BULK_DATA_CATALOG.json`. The record also includes the exact v0.1.1 repository snapshot. Two archives contain the pressure saved-field tiers and one contains the experimental coupled/oracle arrays. The historical 37-tier inventory is separately documented in `data/HISTORICAL_TIER_INDEX.json`; not every historical tier is distributed in this release. No claim of complete native reproduction is made.

All numerical data are generated by simulation or deterministic postprocessing. No generative image model supplies plotted data. Machine-learning assistance in research and drafting is disclosed in the manuscript. Provenance and SHA-256 inventory are provided; see `PROVENANCE.json` and `MANIFEST.json`.

## Citation and rights

Use `CITATION.cff` to cite this versioned model/data release. The accompanying manuscript is not a published article. See `LICENSE` and `LICENSE_STATUS.md`; the MIT licence covers the deposited release contents. Contact: Ehsan Akrami, ehsan.akrami2@mail.dcu.ie. Bugs can be reported through repository issues.
