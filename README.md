# Ozone DLM Pipeline -- Sodankylä

A Dynamic Linear Model (Kalman filter + RTS smoother + MCMC, following
[Laine et al. 2014](https://doi.org/10.5194/acp-14-9707-2014)) estimating
ozone trends at Sodankylä from WOUDC ozonesonde profiles (1988–2026),
corrected for 13 geophysical proxies (solar cycle, QBO, ENSO, polar vortex,
volcanic aerosols, stratospheric temperature, tropopause height, chlorine
loading...).

Four atmospheric layers are analyzed independently: troposphere (0–8 km),
lower stratosphere (8–17 km), mid-stratosphere (17–26 km), and the total
ozone column (from the sonde's own `SondeTotalO3` integration).

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `matplotlib`
- WOUDC ozonesonde profiles, of your own (not bundled in this repo --
  see [Input data](#input-data) below for how to point the code at them)
- The 13 geophysical proxies are included in `proxy/` (see
  [Proxies](#proxies) below)

## Running the pipeline

```bash
cd dlm
python step6_ozone_dlm.py
```

This runs the full pipeline (all 4 layers, with proxies and the 5-level
validation suite) and writes figures to `../output/`.

### Command-line options

| Flag | Default | Effect |
|---|---|---|
| `--no-proxies` | off | Disable the 13 geophysical proxies (raw trend, level+season+AR(1) model only) |
| `--no-validation` | off | Skip the 5-level validation suite (faster, figures only) |
| `--input-dir DIR` | `../input` | Where to read the WOUDC ozonesonde archives from (see [Input data](#input-data)) |
| `--output-dir DIR` | `../output` | Where to write the figures |
| `--n-mcmc N` | 15000 | Post-burn-in MCMC iterations |
| `--n-burnin N` | 3000 | MCMC burn-in iterations |
| `--n-sim N` | 200 | Simulated trajectories used for the 95% confidence band |
| `--aic-threshold X` | 2.0 | AIC gain threshold to keep a proxy in the stepwise selection |

Example -- quick run without proxies or validation, in its own folder:
```bash
python step6_ozone_dlm.py --no-proxies --no-validation --output-dir ../output/no_proxy
```

## What you get

For each of the 4 layers, two figures:

- `dlm_o3_{layer}.png` -- monthly sonde data (points) with the smoothed DLM
  fit (level + seasonal harmonics).
- `dlm_o3_{layer}_slope.png` -- the instantaneous trend $\nu(t)$ in %/decade
  over time, with its 95% MCMC confidence band.

Plus one summary figure comparing all 4 layers:

- `dlm_o3_comparison_layers.png` -- trend ± 95% CI per layer, colored by
  significance (produced locally, not tracked in this repo).

A printed summary table (trend, CI, p-value, selected proxies per layer)
is also written to stdout.

| ![Lower stratosphere fit](output/dlm_o3_lower_strato.png) | ![Lower stratosphere slope](output/dlm_o3_lower_strato_slope.png) |
|---|---|

## Input data

This repo does not bundle the raw WOUDC ozonesonde profiles -- point the
code at wherever you keep them instead. `_sonde_dirs()` in
`dlm/step6_ozone_dlm.py` auto-detects which of two layouts `<input-dir>`
uses:

- **Sodankyla archive layout** -- three chronological subfolders, used
  automatically whenever any of them exists:

  | Subfolder | Coverage | Profiles |
  |---|---|---|
  | `<input-dir>/89-94/woudc/` | 1989--1994 | 382 |
  | `<input-dir>/94-24/woudc/` | 1994--2024 | 1510 |
  | `<input-dir>/24-26/woudc/` | 2024--2026 | 70 |

- **Flat layout** -- used otherwise: any WOUDC extCSV files directly
  inside `<input-dir>/`, no reorganizing needed. This is what a plain
  download for another station looks like (see
  [Using a different station](#using-a-different-station)).

Two ways to set `<input-dir>`:

- Command line: `python step6_ozone_dlm.py --input-dir /path/to/your/data`
- Code default: edit `INPUT_DIR` at the top of `dlm/step6_ozone_dlm.py`
  (defaults to `../input` relative to the `dlm/` folder)

`load_sonde_data()` reads every resolved subfolder and deduplicates dates
that overlap at the archive boundaries.

## Proxies

The `proxy/` folder contains the 13 geophysical proxy time series used by
the model (solar cycle, QBO, ENSO, AO, EHF, SAOD, stratospheric
temperatures, EESC, VPSC, tropopause height...). They are loaded and
standardized by `load_proxies()` in `dlm/step6_ozone_dlm.py`.

**Adding a new proxy is not automatic.** Dropping a new file into `proxy/`
has no effect by itself -- each proxy is loaded by an explicit line inside
`load_proxies()` (e.g. `df["Solar"] = _load_csv_series("mgii.csv", 0,
monthly_idx)`). To add a proxy, two edits are needed in
`dlm/step6_ozone_dlm.py`:

1. Add a loading line for it inside `load_proxies()`.
2. Add its column name to the `PROXY_CANDIDATES` list, so the stepwise AIC
   selection (`select_proxies()`) actually considers it.

## Using a different station

The pipeline is not tied to Sodankyla. Any WOUDC ozonesonde station can
be used with minimal changes.

### Step 1 -- Point the code at your data

WOUDC ozonesonde profiles for any station can be downloaded from
[woudc.org/data/explore.php](https://woudc.org/data/explore.php) ->
*Data -> Ozonesonde*. Filter by station, download the extCSV files into
a single folder, and pass it to the pipeline:

```bash
python step6_ozone_dlm.py --input-dir /path/to/kiruna/sondes
```

No reorganizing is needed: `_sonde_dirs()` only uses the three
chronological subfolders (`89-94/woudc/`, `94-24/woudc/`, `24-26/woudc/`)
for the Sodankyla archive specifically; for any other folder it reads
the extCSV files directly inside it (see [Input data](#input-data)).
`load_sonde_data()` then deduplicates by date as usual.

### Step 2 -- Update the station metadata (optional but recommended)

At the top of `dlm/step6_ozone_dlm.py`, three constants control the
station identity used in console output, figure titles, and output
filenames:

```python
STATION_NAME = "Sodankyla"   # appears in printed banners and figure titles
STATION_CODE = ""            # short tag prefixed to output filenames
LAT, LON     = 67.37, 26.63  # currently cosmetic (printed banner only)
```

Change these for your station, e.g. for Kiruna:

```python
STATION_NAME = "Kiruna"
STATION_CODE = "ki"
LAT, LON     = 67.84, 20.41
```

Figure titles and the printed summary table pick up `STATION_NAME`
automatically. `STATION_CODE`, if set, prefixes every output filename
(`dlm_o3_ki_troposphere.png`, `dlm_o3_ki_comparison_layers.png`, ...) so
that runs for different stations can share the same `--output-dir`
without overwriting each other. Left empty (the Sodankyla default),
filenames are unprefixed, exactly as documented in
[What you get](#what-you-get).

`LAT, LON` are not currently used in any computation -- they only appear
in the printed run banner. Update them anyway so the banner stays
accurate.

### Step 3 -- The tropopause proxy (TP) needs no change

The tropopause height proxy is computed directly from the sonde profiles
by `_tropopause_km()` -- this is **fully automatic**. Each profile's own
temperature/pressure/altitude record is used, so the proxy adapts to the
local tropopause climatology of the new station with no code change.

### Step 4 -- Keep outputs separate

Either rely on `STATION_CODE` (Step 2) or use a station-specific
`--output-dir`, or both:

```bash
python step6_ozone_dlm.py \
  --input-dir /path/to/kiruna/sondes \
  --output-dir ../output/kiruna
```

### Notes

**Proxies.** The 13 geophysical proxies in `proxy/` are regional or
global (solar cycle, QBO, ENSO, AO, EHF, SAOD, EESC, VPSC...) and apply
to any Arctic or sub-Arctic station without modification. For a station
outside the polar vortex's reach, VPSC and VPSC$\times$EESC will be
near-zero year-round and will likely not be selected by the AIC stepwise
procedure -- this is expected, not a bug.

**Data gaps.** Stations with fewer flights per month than Sodankyla's
4--5 will produce wider confidence intervals. The monthly aggregation
(`resample("MS").mean()`) handles this correctly either way.

**Multi-station comparison.** To compare several stations, run the
pipeline once per station with a different `--output-dir` and/or
`STATION_CODE`, then collect the printed summary tables and/or the
`trend_dec`/`trend_p025`/`trend_p975` fields returned by `run_pipeline()`
to plot them together. A dedicated multi-station comparison script is
not included.

## Pipeline structure

| File | Role |
|---|---|
| `dlm/step1_ssm_matrices.py` | State-space model matrices (G, F, Q, R) |
| `dlm/step2_kalman_filter.py` | Forward Kalman filter |
| `dlm/step3_rts_smoother.py` | Backward RTS smoother |
| `dlm/step4_simulation_smoother.py` | Carter-Kohn simulation smoother (trajectory sampling) |
| `dlm/step5_mcmc.py` | Adaptive Metropolis MCMC for hyperparameters |
| `dlm/verif_modele.py` | 5-level validation suite |
| `dlm/step6_ozone_dlm.py` | Orchestration: data loading, proxies, AIC selection, plotting, CLI |
