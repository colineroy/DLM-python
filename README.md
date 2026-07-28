# Ozone DLM Pipeline — Sodankylä

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
- Input data (not included in this folder, resolved via relative paths):
  - `ground/sondes/sondes_data/{89-94,94-24,24-26}/woudc/` — WOUDC ozonesonde profiles
  - `stat/data/proxies/` — pre-downloaded geophysical proxy time series

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
| `--output-dir DIR` | `../output` | Where to write the figures |
| `--n-mcmc N` | 15000 | Post-burn-in MCMC iterations |
| `--n-burnin N` | 3000 | MCMC burn-in iterations |
| `--n-sim N` | 200 | Simulated trajectories used for the 95% confidence band |
| `--aic-threshold X` | 2.0 | AIC gain threshold to keep a proxy in the stepwise selection |

Example — quick run without proxies or validation, in its own folder:
```bash
python step6_ozone_dlm.py --no-proxies --no-validation --output-dir ../output/no_proxy
```

## What you get

For each of the 4 layers, two figures:

- `dlm_o3_{layer}.png` — monthly sonde data (points) with the smoothed DLM
  fit (level + seasonal harmonics).
- `dlm_o3_{layer}_slope.png` — the instantaneous trend $\nu(t)$ in %/decade
  over time, with its 95% MCMC confidence band.

Plus one summary figure comparing all 4 layers:

- `dlm_o3_comparison_layers.png` — trend ± 95% CI per layer, colored by
  significance.

A printed summary table (trend, CI, p-value, selected proxies per layer)
is also written to stdout.

| ![Lower stratosphere fit](output/dlm_o3_lower_strato.png) | ![Lower stratosphere slope](output/dlm_o3_lower_strato_slope.png) |
|---|---|

![4-layer comparison](output/dlm_o3_comparison_layers.png)

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
