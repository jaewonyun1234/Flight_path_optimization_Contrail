# The ISSR model (`contrail_ml`)

This replaces the synthetic Gaussian-blob ISSR field with a machine-learning
model that predicts ice-supersaturated regions (ISSRs) from real weather, and
wraps it in a full MLOps lifecycle. The optimizer is unchanged — the model
plugs in through the same `ISSRField` interface.

## Why this is a real problem

A contrail persists (and warms) only where the air is **ice-supersaturated**:
relative humidity over ice `RHi >= 100 %`. The trouble is that weather models
carry a well-documented **dry bias in RHi near the tropopause** — exactly where
ISSRs form — so raw ERA5 has only a weak ISSR skill (equitable threat score
~0.2–0.4 against in-situ IAGOS measurements). The standard fix is to
**bias-correct RHi against observations**. This project builds a calibrated ML
version of that correction.

## The pipeline

```
IAGOS truth  ─┐
              ├─ collocate ─► training table ─► train ─► register ─► serve ─► monitor
ARCO-ERA5  ──┘                  (parquet)       (MLflow)            (MLIssrField)
```

1. **Features** (`features.py`) — one shared module computes RHi (Murphy–Koop
   ice-saturation), altitude↔pressure, cyclical time encodings, and the
   local↔geo transform. Used identically at train and serve time, so there is
   **no training/serving skew**.
2. **Model** (`model.py`) — `RHiCorrector` is **regime-split**: an XGBoost
   regressor in dry air, a small MLP in the humid regime (split at 85 % RHi,
   where the ISSR signal lives). It predicts the **residual**
   `delta = RHi_IAGOS − RHi_NWP` (the correction), so corrected RHi =
   `RHi_NWP + delta_hat`. A bootstrap ensemble gives a predictive spread; an
   XGBoost head gives `P(ISSR)`.
3. **Calibration** (`calibrate.py`) — isotonic regression makes `P(ISSR)` honest
   (target ECE < 0.05); split-conformal gives a distribution-free RHi interval.
4. **Serving** (`predict.py`, `issr_field.py`) — the model runs over a
   `(lon, lat, pressure)` grid; `MLIssrField` wraps the result and answers the
   optimizer's `is_inside` / `rhi_excess` queries by interpolation. Default
   serving source is the **GFS forecast** (operational); ERA5 replay is
   available for retrospective studies.

### Physics-informed, the legitimate way

The thermodynamics enter as **engineered features** (`rhi`, `e_si(T)`), the
regime split is a **physical boundary**, and outputs are clamped to
thermodynamic plausibility. There is no PDE-residual loss — there is no
governing PDE for this threshold problem, so a PINN would be cargo-culting.

## Reading the comparison table

`train`/`evaluate` print one row per predictor on the temporal holdout:

| column | meaning | better |
|--------|---------|--------|
| `rhi_mae`, `rhi_rmse` | corrected-RHi error vs IAGOS | lower |
| `rhi_bias` | mean error — **the dry-bias number** | nearer 0 |
| `ets` | ISSR equitable threat score | higher |
| `f1`, `roc_auc`, `pr_auc` | ISSR detection skill | higher |
| `brier`, `ece` | probability honesty / calibration | lower |

The model is compared against three baselines it must beat to justify itself:
`raw_era5` (uncorrected), `x_factor` (one global scale), and `quantile_map`
(bivariate T/RHi quantile mapping — the standard statistical correction). The
headline claim is only ever "corrected RHi reduces the dry bias and lifts ISSR
ETS over raw ERA5, with calibrated probabilities" — **reported honestly whatever
the numbers are**.

## Known limitations (stated, not hidden)

- **IAGOS sampling bias** — aircraft avoid deep convection, so the truth set
  under-samples the most intense humidity. Surfaced as an optional
  `sample_weight`; documented, not pretended away.
- **Reanalysis→forecast gap** — we train on ERA5 reanalysis but serve from GFS
  forecast; the domain shift is a real, measured effect (watch it with
  `monitor`).
- **No guarantee of beating quantile mapping** — on real data the gains may be
  modest. The rigorous benchmark + calibration is itself the contribution.
- The bundled `serve-check`/tests use a **synthetic** met cube (clearly
  labelled, guarded) so the seam is exercisable offline. It is never a stand-in
  for real weather.

## MLOps lifecycle

- **Data versioning** — content-addressed parquet + dataset card (+ DVC if set up).
- **Tracking + registry** — MLflow (`mlruns/` by default): params, metrics, the
  comparison table, reliability/scatter plots, git SHA, dataset hash, library
  versions. Model registered as `contrail-issr-rhi-corrector`, moved to Staging.
- **Monitoring** (`monitor.py`) — as new IAGOS arrives, rolling ETS/PR-AUC/ECE
  against fresh truth + per-feature PSI drift, emitting a `retrain_recommended`
  signal. A real ground-truth feedback loop, not a static dashboard.

## Commands

```
python -m contrail_ml build-dataset                  # IAGOS+ERA5 -> versioned parquet
python -m contrail_ml build-dataset --synthetic      # offline fallback table
python -m contrail_ml train --data <parquet>         # CV, fit, calibrate, register
python -m contrail_ml train --synthetic --no-mlflow  # hermetic dry run
python -m contrail_ml evaluate --data <parquet>      # model-vs-baselines table
python -m contrail_ml predict --synthetic            # build + save an ISSR field
python -m contrail_ml serve-check                    # full seam: ML field -> CP-SAT
python -m contrail_ml monitor --reference a.parquet --current b.parquet
```
