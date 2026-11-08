# Integrating the selected pressure multiplier

The selected coefficient is dimensionless. For each element K, multiply both the implemented SH momentum and continuity stabilization parameters by the inferred c_K. This is the specific scalar-length SH implementation described in the manuscript, not every metric-based Shakib formulation.

1. Complete the computed SH predictor using the declared cold-start and boundary-value problem.
2. Construct the continuous_metric24 features from the computed predictor and the same static rectangular Q1 mesh. Pressure feature differences remove the constant datum; no analytic reference enters inference.
3. Apply asinh elementwise, subtract the stored training means, divide by the stored positive scales and clip each standardized feature to [-8,8].
4. For the selected ridge, form the stored affine dot product and intercept. Clip its log output to [log(0.004),log(30)] and exponentiate.
5. Freeze the resulting c_K field for the entire corrected solve, applying it to both SH stabilization channels. Do not update it within Newton or treat it as an accepted-history feedback model.
6. Check the corrected solution's nonlinear acceptance and physical quantities separately. Positive bounded coefficients do not certify solver convergence or accuracy.

```mermaid
flowchart LR
 A[Computed SH predictor] --> B[24 physical features]
 B --> C[Stored asinh normalization and ridge]
 C --> D[Bounded element multiplier]
 D --> E[Freeze both stabilization multipliers]
 E --> F[Corrected flow solve]
```

Charge both flow solves, feature extraction, inference and I/O in a cost comparison. An inference-only timing is not a deployment timing. Missing or rejected predictors remain failures; no analytic-field fallback is allowed. The matched constant may be evaluated without a predictor, but remains a distinct comparison method.

The optional saved-field example computes features and coefficients from one archived predictor; it does not implement or execute the private fluid solver. The experimental shared58/shared62 coefficients belong to a global Oseen operator construction and cannot be substituted into this native local-multiplier interface.
