# Pressure coefficient model card

## Identity and selection

`continuous_affine_ridge` is the selected 25-parameter model. Its 48 normalization constants are part of the model. The complete eight-candidate coefficient file is retained byte-for-byte with its source SHA-256. The matched constant and all six neural candidates are comparison records, not replacements selected on test performance.

## Intended use

Research inference for stationary serial laminar incompressible flow with the declared rectangular Q1 feature and residual-stabilization contract. A converged computed SH predictor supplies the features. No exact solution, family identity or test-error input is used during inference. The raw feature array is transformed with asinh, stored mean/scale and clipping to [-8,8]. The predicted log multiplier is clipped to [log(0.004),log(30)] and exponentiated.

## Training and evaluation populations

45 of 56 planned training cases provide qualified teachers. Four manufactured analytic families were exposed. Validation selects the ridge before the reported fresh-parameter tests. Planned failed/unavailable cases remain listed. Geometric-mean errors have conditional denominators and must not be substituted for all-case reliability.

## Results and limitations

The selected ridge passes 27/32 joint comparisons against SH; its matched constant passes 28/32. Gaussian pressure regressions and unavailable predictors remain. Current-weight smooth-cavity tests and historical cavity/BFS/DFG results have distinct model identities. Do not transfer older physical qualifications to these coefficients. General physical-family, transient, moving-mesh, turbulence and arbitrary mapped/high-order deployment are not qualified. Complete-pipeline comparisons do not establish a joint equal-cost benefit.

## Experimental response operators

The shared58/shared62 files implement a different, global, fixed-mesh Oseen response representation. They are not native local tau regressors. Their known-physics geometry/source setup is required for deployment; an NPZ coefficient file alone is not a complete solver. The expanded primary fails additional full57 pressure/velocity mixture tests at all thirteen fitting settings. Neither the trace-enrichment oracle nor numerical fitting admission is a reliable coupled learned model.

## Reuse

Check the model identifier, feature variant, array ordering and checksum. Retain bounds and normalization. A failed predictor is a failed deployment position; silently substituting analytic features or a successful comparator changes the experiment. Redistribution terms remain pending. The private solver is excluded.
