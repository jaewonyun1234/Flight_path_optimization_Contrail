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

---

# Real flights for `contrail_flights` (OpenSky)

`contrail_flights` is an **independent** sibling package (it shares no imports
with `contrail_ml`). It turns real flown **historical** ADS-B tracks into the
same `contrail_env.Flight` objects the synthetic generator produces, so the
optimizer, solvers, and GUI map consume them unchanged. It is opt-in via
`flight_source="real"`; the default stays synthetic. Requires the `[flights]`
extra (`pyopensky`, `pandas`, `pyarrow`); the hermetic tests need none of it.

> **Honest naming.** OpenSky gives real flown *historical* traffic (actual ADS-B
> tracks), **not** published schedules or planned flights. The accurate claim is
> "demonstrated on real historical European traffic" — the same evaluation method
> used in real contrail-avoidance trials (Google / American Airlines) — not
> "schedule optimization".

## OpenSky access

| Path | Window | Auth | Use |
|------|--------|------|-----|
| Public REST API | recent rolling ~1–2 h | none (rate-limited) | small/recent pulls |
| Research / Trino | arbitrary history, whole regions | account required | bulk historical |

1. **Recent/light:** the unauthenticated REST API covers only a short rolling
   window and is heavily rate-limited — fine for a quick demo, not for history.
2. **Bulk historical:** apply for an OpenSky **research/Trino account** at
   <https://opensky-network.org> (free for academic use). Put credentials in the
   environment — never in the repo:
   ```
   export OPENSKY_USERNAME=...
   export OPENSKY_PASSWORD=...
   ```
3. Pulled tracks are cached as parquet (keyed by bbox + time window) under
   `cache_dir`, since bulk queries are rate-limited/credit-metered.

**No fabrication.** If `pyopensky` is missing, credentials are absent, or the
query fails/returns empty, `OpenSkyClient` raises `OpenSkyUnavailableError` with
an actionable message — it never returns invented or zero traffic.

## Aircraft type → performance profile

The optimizer needs an `Aircraft` performance model per flight. OpenSky's free
aircraft-database CSV maps `icao24` → ICAO type designator; we map that type to
the **nearest existing** profile via an explicit table
(`reduce_to_flight.AIRCRAFT_TYPE_TO_FACTORY`):

| ICAO type | Profile | Note |
|-----------|---------|------|
| A319 / A320 / A321 | `a320_like` | linear narrowbody surrogate |
| B737 / B738 | `a320_like` | same class |
| *(anything else)* | `a320_like` (default) | documented approximation |

**Limitation:** we do **not** synthesize a new performance model per type — every
mapped type currently points at the one `a320_like` linear surrogate. The table
is the single place to add real BADA-class profiles later.

## Baseline = the real flown profile

For real flights the `baseline` is, by preference, the **observed** altitude
profile: the track's barometric altitude is snapped to the RVSM flight-level grid
and resampled into `AltitudeSegment`s. Only when altitude data is too sparse/noisy
do we fall back to the synthetic fuel-optimal `build_baseline_profile`, and
`build_dataset` logs how many flights hit that fallback.

## Config keys

`contrail_flights.config.FlightsConfig`, overridable from `CONTRAIL_FLIGHTS_*`
env vars (plus `OPENSKY_USERNAME`/`OPENSKY_PASSWORD`). Key ones: the lon/lat
`bbox`, `start_time`/`end_time`, `min_track_points`/`min_track_km`,
`snapshot_window_s`, `cache_dir`, and the anchor (`origin_lat`, `origin_lon`,
which **must** match `MLConfig` so flights and the predicted field align).
