# Data sources for `contrail_ml`

The ISSR model trains on three free-to-academic data sources. None of them are
committed to the repo, and **no credentials live in the codebase** — paths and
keys come from the environment (see `.env.example`). The hermetic tests use the
guarded `synthetic_fallback` and need none of this.

| Source | Role | Cost | Access |
|--------|------|------|--------|
| **IAGOS** | ground-truth in-situ humidity (labels) | free, registration | portal sign-up |
| **ARCO-ERA5** | reanalysis weather (training inputs) | free | public GCS bucket, no creds |
| **GFS** | forecast weather (serving inputs) | free, public domain | NOAA, via pycontrails |
| ECMWF HRES | best forecast (optional, not used) | paid/licensed | — |

## IAGOS (the labels)

IAGOS (In-service Aircraft for a Global Observing System,
<https://www.iagos.org>) equips airliners with research humidity sensors. Its
measured relative-humidity-over-ice (RHi) near the tropopause is the truth this
project trains against.

1. Register (free) on the IAGOS data portal and download the per-flight NetCDF
   files for the years/region you want.
2. Put them in a folder and point the loader at it:
   ```
   export CONTRAIL_ML_IAGOS_DIR=/path/to/iagos/netcdf
   ```
3. `contrail_ml/data/iagos.py` recomputes RHi from the measured T, water-vapour
   and pressure with the SAME thermodynamics as the features (no label skew).

**Variable names differ between IAGOS products** (IAGOS-CORE, MOZAIC, CARIBIC).
The defaults in `IAGOS_VARIABLES` target IAGOS-CORE; override per product via
`cfg.extra["iagos_variables"]`. Verify the names in your NetCDF before a real run.

## ARCO-ERA5 (the training inputs)

ERA5 is ECMWF's reanalysis — the best reconstruction of *past* weather. We read
the **ARCO-ERA5** zarr mirror on a public Google Cloud bucket through
pycontrails, which needs **no Copernicus CDS account**. `era5.py` pulls the
fields, `collocate.py` interpolates them onto the IAGOS waypoints.

Requires the `[ml]` extra (`pycontrails`, `xarray`, `zarr`, `gcsfs`).

## GFS (the serving inputs)

At serve time you need where ISSR *will* be, so the operational input is NOAA's
**GFS forecast** — free, public-domain, refreshed every 6 h, delivered as GRIB
(hence `cfgrib`). `gfs.py` loads it via pycontrails' `GFSForecast`.

## Config keys

All configuration is `contrail_ml.config.MLConfig`, overridable from
`CONTRAIL_ML_*` environment variables (see `.env.example`) or a YAML file. Key
ones: `iagos_dir`, `mlflow_tracking_uri`, `pressure_levels_hpa`, the lon/lat
box, `train_years`/`test_years`, and the geographic anchor (`origin_lat`,
`origin_lon`).

## Reproducibility

`build-dataset` writes a **content-addressed** parquet plus a `dataset_card.json`
(row count, time/region coverage, class balance, dry-bias). Every training run
records that dataset hash, so any reported number is traceable to an exact table.
If DVC is installed the parquet is `dvc add`-ed automatically.
