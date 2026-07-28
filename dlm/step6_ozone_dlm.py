"""
=============================================================================
STEP 6 / 6  --  Full ozone DLM pipeline, Sodankyla
=============================================================================

Layers (4, analogous to the CH4 approach: 3 partial layers + 1 "total
column") -- ALL derived from WOUDC ozonesondes 1988-2026 (FTIR, initially
tried for the total column, was abandoned: only 72 months, 2012-2021,
gave an unusable +/-186%/dec trend):
    troposphere   0-8 km    -- local pollution, comparison only
    lower_strato  8-17 km   -- polar vortex entry
    mid_strato    17-26 km  -- polar vortex core
    total         total column measured by the sonde itself
                  (#FLIGHT_SUMMARY / SondeTotalO3 -- integration of the
                  profile up to burst altitude, plus a residual
                  correction above it, NOT a simple sum of our 3
                  layers capped at 26 km)

Sondes capped at 26 km (SONDE_CAP_KM) for the partial layers: above
that, ECC balloons become unreliable (burst ~30-35 km, measurement
increasingly imprecise at decreasing pressure).

Proxies (13, see ../proxy/, already downloaded), EXCEPT TP which is
recomputed by us:
    Solar (Mg II), QBO30, QBO10, ENSO (MEI), AO, EHF, SAOD,
    T_LS, T_MS, EESC, VPSC, VPSC_EESC
    TP: WMO thermal tropopause height, computed directly on each sonde
        profile via get_trop_height() (CH4/scripts/tropo_height.py,
        the same algorithm already validated for the tropo/strato
        separation in the CH4 pipeline) rather than the external ERA5
        reanalysis -- consistent with the instrument, and covers
        1988-2026 instead of 1989-2024.
Stepwise AIC selection, threshold 2 units (consistent with the
Sutherland et al. 2023 calibration used for CH4 -- stat/DLM used an
uncalibrated threshold of 1.0).
"""

import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from step1_ssm_matrices import make_G, make_F, make_Q, make_R, N_STATE, RHO, FREQS
from step2_kalman_filter import kalman_filter
from step3_rts_smoother import kalman_smoother, rolling_trend_from_level
from step4_simulation_smoother import simulation_smoother, rolling_trends_mcmc
from step5_mcmc import make_prior, run_mcmc, mcmc_diagnostics
from verif_modele import run_full_validation

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "CH4" / "scripts"))
from tropo_height import get_trop_height  # noqa: E402 (already validated on CH4)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STATION   = "Sodankyla"
LAT, LON  = 67.37, 26.63

HERE      = Path(__file__).parent
PROJ_ROOT = HERE.parent.parent            # C:\Users\royc\OneDrive\NDACC
# The 3 WOUDC folders covering the whole Sodankyla archive:
# 89-94 (NOGDB, 382 profiles), 94-24 (1510 profiles), 24-26 (SHARP, 70 profiles).
# stat/data/sondes/woudc/ is only a copy of 94-24 -- we read directly from
# ground/sondes/, which has all 3 periods.
SONDE_DIRS = [
    PROJ_ROOT / "ground" / "sondes" / "sondes_data" / "89-94" / "woudc",
    PROJ_ROOT / "ground" / "sondes" / "sondes_data" / "94-24" / "woudc",
    PROJ_ROOT / "ground" / "sondes" / "sondes_data" / "24-26" / "woudc",
]
PROXY_DIR = HERE.parent / "proxy"

OUTPUT_DIR = HERE.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

LAYERS = {
    "troposphere":  (0,  8),
    "lower_strato": (8,  17),
    "mid_strato":   (17, 26),
}
SONDE_CAP_KM = 26.0

AIC_THRESHOLD = 2.0   # calibrated on Sutherland et al. (2023), see the CH4 report
USE_PROXIES   = True

PROXY_CANDIDATES = ["Solar", "QBO30", "QBO10", "ENSO", "AO", "EHF", "SAOD",
                    "TP", "T_LS", "T_MS", "EESC", "VPSC", "VPSC_EESC"]

N_MCMC   = 15000
N_BURNIN = 3000
N_SIM    = 200
WINDOW_YR = 20   # Nilsen et al. (2024): slow post-Montreal recovery


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOADING WOUDC SONDES (3 partial layers)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_woudc_extcsv(filepath):
    """Parses a WOUDC extCSV file (#PROFILE section) -- ported from
    stat/DLM/DLM_nilsen.py, already validated on the 1510 profiles 1994-2024."""
    date_str = filepath.stem[:8]
    date = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    data_start = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if "#PROFILE" in line.upper():
                data_start = i + 1
                break
    if data_start is None:
        raise ValueError("PROFILE section not found")
    raw = pd.read_csv(filepath, skiprows=data_start, comment="*",
                      on_bad_lines="skip", escapechar="\\")
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {"Pressure": "P_hPa", "O3PartialPressure": "O3_mPa",
               "Temperature": "T_C", "GPHeight": "Z_m",
               "Altitude": "Z_m", "Height": "Z_m"}
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "Z_m" in raw.columns:
        raw["Z_km"] = raw["Z_m"] / 1000.0
    elif "P_hPa" in raw.columns:
        T_K = (raw["T_C"].values + 273.15 if "T_C" in raw.columns
               else np.full(len(raw), 250.0))
        raw["Z_km"] = (8.314 * np.nanmean(T_K) / (0.029 * 9.81)
                       * np.log(1013.25 / raw["P_hPa"].values) / 1000)
    else:
        raise ValueError("Altitude not found")
    if "T_C" in raw.columns:
        raw["T_K"] = raw["T_C"] + 273.15
    raw["date"] = date
    return raw[raw["Z_km"] <= SONDE_CAP_KM].dropna(subset=["Z_km", "O3_mPa"])


def _parse_woudc_total_o3(filepath):
    """Extracts SondeTotalO3 from the #FLIGHT_SUMMARY section (integration
    of the profile by the sonde itself up to burst, plus a residual
    correction above it -- ported from CH4/scripts/dop/extract_woudc.py)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip() == "#FLIGHT_SUMMARY":
                next(fh)  # header
                data = next(fh).strip()
                parts = data.split(",")
                if len(parts) >= 3:
                    val = parts[2].strip()
                    if val.lower() not in ("nan", "9999", ""):
                        return float(val)
                break
    return np.nan


def _tropopause_km(profile):
    """WMO thermal tropopause height computed directly on the sonde
    profile (get_trop_height, the same algorithm as CH4/scripts/
    tropo_height.py already validated for FTIR). Sonde profiles are
    already in surface->top order (balloon ascent), so no reversal is
    needed, unlike the FTIR HDF4 files (top->surface)."""
    p = profile.sort_values("Z_km")
    if len(p) < 5:
        return np.nan
    GH = p["Z_km"].values * 1000.0
    P  = p["P_hPa"].values * 100.0
    AT = p["T_K"].values if "T_K" in p.columns else None
    if AT is None or np.isnan(AT).all():
        return np.nan
    try:
        ztrop_m, _ = get_trop_height(AT, GH, P)
        return ztrop_m / 1000.0
    except (ValueError, IndexError):
        return np.nan


def _integrate_km(profile, z0, z1):
    """Partial O3 column in DU between z0 and z1 km (barometric
    integration, constant 269.6 = standard DU conversion)."""
    layer = profile[(profile["Z_km"] >= z0) & (profile["Z_km"] <= z1)].copy()
    if len(layer) < 3:
        return np.nan
    layer = layer.sort_values("Z_km")
    t_k = layer["T_K"].values if "T_K" in layer.columns else np.full(len(layer), 230.0)
    col = 269.6 * np.trapezoid(layer["O3_mPa"].values / t_k, layer["Z_km"].values)
    return round(abs(col), 2)


def load_sonde_data(data_dirs: list = SONDE_DIRS) -> pd.DataFrame:
    """Loads WOUDC profiles from the 3 periods (89-94, 94-24, 24-26),
    integrates per layer -> daily DataFrame over 1988-2026."""
    files = []
    for d in data_dirs:
        found = sorted(Path(d).glob("*.csv"))
        print(f"    {d.parent.name} : {len(found)} profiles")
        files.extend(found)
    if not files:
        raise FileNotFoundError(f"No WOUDC profiles in {data_dirs}")

    # The 3 folders overlap at their boundaries (12 shared dates
    # 89-94/94-24 late 1994, 30 shared dates 94-24/24-26 in 2024) --
    # the same sonde launch is sometimes archived twice by different
    # WOUDC batches. Deduplicate by date (keep the first occurrence, in
    # the folders' chronological order).
    records = []
    n_fail = 0
    seen_dates = set()
    n_dup = 0
    for f in files:
        try:
            profile = _parse_woudc_extcsv(f)
            date = profile["date"].iloc[0]
            if date in seen_dates:
                n_dup += 1
                continue
            seen_dates.add(date)
            row = {"date": date}
            for layer, (z0, z1) in LAYERS.items():
                row[f"O3_{layer}"] = _integrate_km(profile, z0, z1)
            row["O3_total"] = _parse_woudc_total_o3(f)
            row["tropopause_km"] = _tropopause_km(profile)
            records.append(row)
        except Exception:
            n_fail += 1

    df = (pd.DataFrame(records)
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .dropna(how="all", subset=[f"O3_{l}" for l in LAYERS])
          .sort_values("date")
          .set_index("date"))
    n_tp = df["tropopause_km"].notna().sum()
    n_tot = df["O3_total"].notna().sum()
    print(f"  [OK] {len(df)} sonde profiles ({df.index.min().year}-"
          f"{df.index.max().year}), {n_dup} duplicates discarded, "
          f"{n_fail} read failures")
    print(f"       {n_tot} total columns (SondeTotalO3), "
          f"{n_tp} tropopauses detected "
          f"(mean {df['tropopause_km'].mean():.1f} km)")
    return df


def sonde_to_monthly(df_sonde: pd.DataFrame) -> pd.DataFrame:
    """Monthly mean per layer, normalized by the period mean (same
    convention as the CH4 pipeline -- no deseasonalization here, the DLM
    models the seasonal cycle itself via the harmonics)."""
    monthly = df_sonde.resample("MS").mean()
    monthly.index.freq = None
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROXIES (13 series, already downloaded into proxy/, except TP)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_csv_series(fname, col_idx, monthly_idx):
    fpath = PROXY_DIR / fname
    if not fpath.exists():
        return None
    raw = pd.read_csv(fpath, parse_dates=["date"], index_col="date")
    s = raw.iloc[:, col_idx]
    s = s.resample("MS").mean()
    return s.reindex(monthly_idx).interpolate(limit=3)


def _load_eesc(monthly_idx):
    fpath = PROXY_DIR / "odgi_table2.csv"
    raw = pd.read_csv(fpath)
    raw.columns = [c.strip() for c in raw.columns]
    eesc_col = [c for c in raw.columns if "EESC SUM" in c and "new" in c][0]
    data = raw[["Year", eesc_col]].dropna().astype(float)
    years, vals = data["Year"].values, data[eesc_col].values
    start = f"{int(years[0])}-01-01"
    end   = f"{int(years[-1]) + 1}-12-31"
    daily_idx = pd.date_range(start, end, freq="D")
    daily_vals = np.interp(np.arange(len(daily_idx)),
                          (years - years[0]) * 365.25, vals)
    eesc_daily = pd.Series(daily_vals, index=daily_idx)
    eesc_monthly = eesc_daily.resample("MS").mean()
    return eesc_monthly.reindex(monthly_idx).interpolate(limit=6)


def _load_vpsc(monthly_idx):
    fpath = PROXY_DIR / "vpsc-370-550_1994-2024.txt"
    raw = pd.read_csv(fpath, sep=r"\s+")
    dates = pd.to_datetime(raw[["year", "mon", "day"]]
                           .rename(columns={"mon": "month", "day": "day"}))
    vpsc_s = pd.Series(raw["NAT_N"].values, index=dates)
    vpsc_m = vpsc_s.resample("MS").mean()
    return vpsc_m.reindex(monthly_idx).interpolate(limit=3)


def load_proxies(monthly_idx: pd.DatetimeIndex,
                 tp_sonde: pd.Series | None = None) -> pd.DataFrame:
    """Loads the 13 proxies (proxy/, except TP), interpolates
    short gaps, standardizes (mean=0, std=1).

    tp_sonde : monthly tropopause height computed on the sondes
        (get_trop_height, see _tropopause_km) -- if provided, replaces
        the external TP proxy (ERA5) with our own computation, consistent
        with the instrument and covering 1988-2026 instead of 1989-2024."""
    print("\n[2] Loading proxies...")
    df = pd.DataFrame(index=monthly_idx)

    df["Solar"] = _load_csv_series("mgii.csv", 0, monthly_idx)

    qbo_raw = pd.read_csv(PROXY_DIR / "qbo_pc.csv", parse_dates=["date"],
                         index_col="date").resample("MS").mean()
    df["QBO30"] = qbo_raw.iloc[:, 0].reindex(monthly_idx).interpolate(limit=3)
    df["QBO10"] = qbo_raw.iloc[:, 1].reindex(monthly_idx).interpolate(limit=3)

    df["ENSO"] = _load_csv_series("mei.csv", 0, monthly_idx)
    df["AO"]   = _load_csv_series("ao.csv", 0, monthly_idx)
    df["EHF"]  = _load_csv_series("ehf.csv", 0, monthly_idx)
    df["SAOD"] = _load_csv_series("saod.csv", 0, monthly_idx)
    if tp_sonde is not None:
        df["TP"] = tp_sonde.reindex(monthly_idx).interpolate(limit=3)
        print("    [OK] TP computed from the sondes (get_trop_height)")
    else:
        df["TP"] = _load_csv_series("tropopause_era5.csv", 0, monthly_idx)
    df["T_LS"] = _load_csv_series("temp_strato_ls.csv", 0, monthly_idx)
    df["T_MS"] = _load_csv_series("temp_strato_ms.csv", 0, monthly_idx)

    eesc = _load_eesc(monthly_idx)
    vpsc = _load_vpsc(monthly_idx)
    df["EESC"] = eesc
    df["VPSC"] = vpsc
    df["VPSC_EESC"] = vpsc * eesc

    # Deseasonalize everything EXCEPT VPSC/VPSC_EESC (their seasonal cycle
    # is the physical signal itself -- polar stratospheric clouds only
    # exist in polar winter).
    doy = df.index.day_of_year
    keep_raw = {"VPSC", "VPSC_EESC"}
    for col in df.columns:
        if col in keep_raw or df[col].notna().sum() < 24:
            continue
        clim = df[col].groupby(doy).transform("mean")
        df[col] = df[col] - clim

    df = (df - df.mean()) / df.std()
    loaded = df.notna().all(axis=1).sum()
    n_before = len(df)
    # Some proxies (e.g. TP, 1989-2024) do not cover the whole extended
    # sonde series (1988-2026): beyond the interpolation limit, fill with
    # 0 (= standardized mean) rather than leaving NaN, which would crash
    # the Kalman filter (v(t) = y(t) - F.x - proxy.beta becomes NaN if a
    # single proxy is NaN, even if y(t) is valid).
    n_filled = df.isna().sum().sum()
    df = df.fillna(0.0)
    print(f"  [OK] {len(df.columns)} proxies | {loaded}/{n_before} complete months "
          f"({n_filled} out-of-coverage values filled with 0)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STEPWISE AIC PROXY SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def aic(y, t, X, selected):
    """OLS AIC: n*log(RSS/n) + 2k. y, t aligned; X = proxy DataFrame."""
    cols = [np.ones_like(t), t]
    if selected:
        cols += [X[c].values for c in selected]
    A = np.column_stack(cols)
    valid = ~np.isnan(y) & np.all(~np.isnan(A), axis=1)
    coef, *_ = np.linalg.lstsq(A[valid], y[valid], rcond=None)
    resid = y[valid] - A[valid] @ coef
    n = valid.sum()
    k = A.shape[1]
    return n * np.log(np.sum(resid**2) / n) + 2 * k


def select_proxies(y: pd.Series, proxy_df: pd.DataFrame,
                   candidates: list = PROXY_CANDIDATES,
                   threshold: float = AIC_THRESHOLD,
                   verbose: bool = True) -> list:
    """Stepwise forward selection by AIC (decision threshold =
    `threshold` units, calibrated on Sutherland et al. 2023)."""
    t = np.arange(len(y), dtype=float)
    y_v = y.values
    available = [c for c in candidates if c in proxy_df.columns]
    selected = []
    aic_base = aic(y_v, t, proxy_df, selected)
    if verbose:
        print(f"    Base AIC (1 + t): {aic_base:.1f}")

    remaining = list(available)
    while remaining:
        gains = {}
        for c in remaining:
            aic_c = aic(y_v, t, proxy_df, selected + [c])
            gains[c] = aic_base - aic_c
        best_c = max(gains, key=gains.get)
        best_gain = gains[best_c]
        if best_gain >= threshold:
            selected.append(best_c)
            aic_base -= best_gain
            remaining.remove(best_c)
            if verbose:
                print(f"      + {best_c} : AIC gain = {best_gain:.2f}")
        else:
            if verbose:
                print(f"      Stop: best gain = {best_gain:.2f} < {threshold}")
            break
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-LAYER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_dlm_layer(y_raw: pd.Series, proxy_df: pd.DataFrame | None,
                  layer_name: str, verbose: bool = True,
                  use_proxies: bool = USE_PROXIES) -> dict:
    print(f"\n{'='*60}")
    print(f"  Layer: {layer_name.upper().replace('_', ' ')} OZONE")
    print(f"{'='*60}")

    y_idx = y_raw.index
    y_mean = y_raw.mean()
    y_norm = (y_raw / y_mean).values

    selected, exog, beta = [], None, None
    if use_proxies and proxy_df is not None:
        common = y_idx.intersection(proxy_df.index)
        proxy_aligned = proxy_df.reindex(y_idx)
        print(f"  [1] AIC proxy selection (threshold={AIC_THRESHOLD})...")
        selected = select_proxies(pd.Series(y_norm, index=y_idx), proxy_aligned,
                                  threshold=AIC_THRESHOLD)
        if selected:
            exog = proxy_aligned[selected].values
        print(f"  -> proxies retained: {selected if selected else 'none'}")

    print(f"  [2] Prior + MCMC ({N_MCMC} iter, {N_BURNIN} burn-in)...")
    prior = make_prior(y_norm)
    mcmc_r = run_mcmc(y_norm, prior, n_iter=N_MCMC, n_burnin=N_BURNIN,
                      proxies=exog, beta=beta, verbose=verbose)

    sigma_trend = np.median(mcmc_r["sigma_trend"])
    sigma_seas  = np.median(mcmc_r["sigma_seas"])
    sigma_obs   = np.median(mcmc_r["sigma_obs"])
    sigma_ar    = np.median(mcmc_r["sigma_ar"])
    rho_est     = np.median(mcmc_r["rho"])
    beta_est    = mcmc_r["beta"]

    kf = kalman_filter(y_norm, sigma_obs, sigma_trend, sigma_seas, sigma_ar,
                       exog, beta_est, rho=rho_est)
    ks = kalman_smoother(kf, rho=rho_est)
    sim = simulation_smoother(ks, n_samples=N_SIM)

    t_arr = np.arange(len(y_norm), dtype=float)
    trends = []
    for k in range(sim["level_samp"].shape[0]):
        mu = sim["level_samp"][k]
        valid = ~np.isnan(mu)
        if valid.sum() > 10:
            A = np.column_stack([np.ones(valid.sum()), t_arr[valid]])
            c, *_ = np.linalg.lstsq(A, mu[valid], rcond=None)
            trends.append(c[1] * 120 * 100)
    trends = np.array(trends)
    trend_dec  = np.median(trends)
    trend_p025 = np.percentile(trends, 2.5)
    trend_p975 = np.percentile(trends, 97.5)
    trend_std  = np.std(trends)
    from scipy import stats as _stats
    z = trend_dec / trend_std if trend_std > 0 else 0
    p_value = 2 * (1 - _stats.norm.cdf(abs(z)))

    print(f"\n  -> DLM O3 trend [{layer_name}]: {trend_dec:+.2f} "
          f"+/- {(trend_p975-trend_p025)/2:.2f} %/dec  (p={p_value:.4f})")

    return {
        "layer": layer_name, "y_idx": y_idx, "y_vals": y_norm,
        "layer_mean": y_mean,
        "sigma_trend": sigma_trend, "sigma_seas": sigma_seas,
        "sigma_obs": sigma_obs, "sigma_ar": sigma_ar, "rho": rho_est,
        "mcmc": mcmc_r, "kf": kf, "ks": ks, "sim": sim,
        "trend_dec": trend_dec, "trend_p025": trend_p025,
        "trend_p975": trend_p975, "trend_std": trend_std, "p_value": p_value,
        "proxies_sel": selected, "exog": exog, "beta": beta_est,
    }


COLOR_DATA  = "#4fc3f7"
COLOR_FIT   = "#ff6b6b"
COLOR_BAND  = "#66bb6a"


def plot_layer_results(res: dict, output_dir: Path = OUTPUT_DIR):
    """
    2 figures per layer:
      1. dlm_o3_{layer}.png: observed data (absolute DU units) +
         smoothed DLM fit (level + harmonics).
      2. dlm_o3_{layer}_slope.png: instantaneous slope nu(t) in
         %/decade, MCMC confidence band (2.5-97.5th percentile over the
         simulated trajectories), overall trend for reference.
    """
    # verif_modele._setup_dark() mutates GLOBAL matplotlib rcParams
    # (axes.facecolor, etc.) without ever resetting them -- if Level 5
    # of a previous layer just ran, these figures would inherit a dark
    # background. Reset to defaults before every result figure, to
    # guarantee a white background regardless of call order with
    # run_full_validation().
    plt.rcdefaults()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    layer = res["layer"]
    dates = res["y_idx"]
    y_abs = res["y_vals"] * res["layer_mean"]
    ks = res["ks"]
    sim = res["sim"]

    # Signal reconstruction (level + cosine harmonics, in absolute
    # units) from the smoothed states -- same indices as make_F().
    from step1_ssm_matrices import FREQS as _FREQS
    harm_idx = [2 + 2*k for k in range(len(_FREQS))]
    fit_norm = ks["x_smooth"][:, 0] + ks["x_smooth"][:, harm_idx].sum(axis=1)
    fit_abs = fit_norm * res["layer_mean"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, y_abs, "o", ms=3, alpha=0.4, color=COLOR_DATA, label="Sondes (monthly)")
    ax.plot(dates, fit_abs, lw=1.8, color=COLOR_FIT, label="DLM (level + season)")
    ax.set_ylabel("O$_3$ (DU)", fontsize=11)
    ax.set_title(f"Ozone {layer.replace('_',' ')} -- Sodankyla (sondes 1988-2026)",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.tight_layout()
    fig.savefig(output_dir / f"dlm_o3_{layer}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Instantaneous slope nu(t) with MCMC band ─────────────────────────
    slope_samp = sim["slope_samp"]   # (M, n) %/dec
    slope_p025 = np.percentile(slope_samp, 2.5, axis=0)
    slope_p975 = np.percentile(slope_samp, 97.5, axis=0)
    slope_med  = np.median(slope_samp, axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates, slope_p025, slope_p975, alpha=0.25, color=COLOR_BAND,
                    label="95% CI (MCMC)")
    ax.plot(dates, slope_med, lw=1.8, color=COLOR_BAND, label="$\\nu(t)$ median")
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.axhline(res["trend_dec"], color=COLOR_FIT, lw=1.2, ls=":",
              label=f"Overall trend = {res['trend_dec']:+.2f} %/dec")
    ax.set_ylabel("$\\nu(t)$ [%/decade]", fontsize=11)
    ax.set_title(f"Instantaneous slope -- Ozone {layer.replace('_',' ')}",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.tight_layout()
    fig.savefig(output_dir / f"dlm_o3_{layer}_slope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> Figures: dlm_o3_{layer}.png, dlm_o3_{layer}_slope.png (in {output_dir}/)")


def plot_layers_comparison(results: dict, output_dir: Path = OUTPUT_DIR):
    """Summary (forest-plot style) figure: trend +/- 95% CI for the 4
    layers, color-coded by significance."""
    plt.rcdefaults()  # see plot_layer_results: avoids inheriting the dark theme
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    layers = [l for l in results if not l.endswith("_validation")]
    labels = [l.replace("_", " ").title() for l in layers]
    trends = [results[l]["trend_dec"] for l in layers]
    lo = [results[l]["trend_p025"] for l in layers]
    hi = [results[l]["trend_p975"] for l in layers]
    pvals = [results[l]["p_value"] for l in layers]
    err_lo = [t - l for t, l in zip(trends, lo)]
    err_hi = [h - t for t, h in zip(trends, hi)]

    colors = [COLOR_FIT if p < 0.05 else "#999999" for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(layers))
    # ecolor does not accept a per-point list of colors -- one error bar
    # at a time so the color can vary by significance.
    for i in range(len(layers)):
        ax.errorbar([trends[i]], [y_pos[i]], xerr=[[err_lo[i]], [err_hi[i]]],
                   fmt="o", ms=8, color="black", ecolor=colors[i],
                   elinewidth=2.5, capsize=5)
    for i, (t, p) in enumerate(zip(trends, pvals)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(hi[i] + 0.3, y_pos[i], f"{t:+.2f} ({sig})", va="center", fontsize=9)
    ax.axvline(0, color="#888", lw=1, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("O$_3$ trend [%/decade]", fontsize=11)
    ax.set_title("DLM trends by layer -- Ozone Sodankyla (1988-2026)",
                fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dlm_o3_comparison_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figure: dlm_o3_comparison_layers.png (in {output_dir}/)")


def print_summary(results: dict):
    print("\n" + "="*72)
    print("  DLM OZONE RESULTS (Laine 2014 + Nilsen 2024) - Sodankyla")
    print("="*72)
    for layer, res in results.items():
        if layer.endswith("_validation"):
            continue
        sig = "***" if res["p_value"] < 0.001 else (
              "**" if res["p_value"] < 0.01 else (
              "*" if res["p_value"] < 0.05 else "n.s."))
        print(f"  {layer:<14} {res['trend_dec']:>+8.2f} +/- "
              f"{(res['trend_p975']-res['trend_p025'])/2:>5.2f} %/dec  "
              f"p={res['p_value']:.4f} {sig}")
        print(f"                 {len(res['y_vals'])} months | "
              f"{len(res['proxies_sel'])} proxies: "
              f"{', '.join(res['proxies_sel']) if res['proxies_sel'] else 'none'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(run_validation: bool = True, use_proxies: bool = USE_PROXIES,
                 output_dir: Path = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("="*60)
    print(f"  DLM OZONE - {STATION} {LAT}N {LON}E")
    print(f"  WOUDC sondes 1988-2026 (3 partial layers + total column)")
    print(f"  Proxies: {'yes' if use_proxies else 'no'} | Output: {output_dir}")
    print("="*60)

    print("\n[1] Loading WOUDC sondes...")
    df_sonde_daily = load_sonde_data()
    monthly_sonde = sonde_to_monthly(df_sonde_daily)

    proxy_df = (load_proxies(monthly_sonde.index, tp_sonde=monthly_sonde["tropopause_km"])
               if use_proxies else None)

    results = {}
    for layer in list(LAYERS) + ["total"]:
        y_raw = monthly_sonde[f"O3_{layer}"].dropna()

        res = run_dlm_layer(y_raw, proxy_df, layer, use_proxies=use_proxies)
        results[layer] = res
        plot_layer_results(res, output_dir=output_dir)

        if run_validation:
            print(f"\n[Validation] 5 diagnostic levels ({layer})...")
            val = run_full_validation(
                y=res["y_vals"], dates=res["y_idx"],
                sigma_trend=res["sigma_trend"], sigma_seas=res["sigma_seas"],
                sigma_obs=res["sigma_obs"], sigma_ar=res["sigma_ar"],
                rho=res["rho"], trend_dec=res["trend_dec"],
                mcmc_result=res["mcmc"],
                proxies=res.get("exog"), beta=res.get("beta"),
            )
            results[layer + "_validation"] = val

    plot_layers_comparison(results, output_dir=output_dir)
    print_summary(results)
    print(f"\n  [OK] Outputs in '{output_dir}/'")
    return results


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Ozone DLM pipeline (Sodankyla) -- WOUDC sondes 1988-2026, "
                    "4 layers (troposphere, lower/mid stratosphere, total column).")
    p.add_argument("--no-proxies", action="store_true",
                   help="Disable the 13 geophysical proxies (raw trend, "
                       "level+season+AR(1) model only).")
    p.add_argument("--no-validation", action="store_true",
                   help="Disable the 5 validation levels (faster, "
                       "produces only the trend figures).")
    p.add_argument("--output-dir", type=str, default=None,
                   help=f"Output directory for the figures (default: {OUTPUT_DIR}).")
    p.add_argument("--n-mcmc", type=int, default=N_MCMC,
                   help=f"Post-burn-in MCMC iterations (default: {N_MCMC}).")
    p.add_argument("--n-burnin", type=int, default=N_BURNIN,
                   help=f"MCMC burn-in iterations (default: {N_BURNIN}).")
    p.add_argument("--n-sim", type=int, default=N_SIM,
                   help=f"Simulated trajectories for the 95%% CI (default: {N_SIM}).")
    p.add_argument("--aic-threshold", type=float, default=AIC_THRESHOLD,
                   help=f"AIC gain threshold to retain a proxy (default: {AIC_THRESHOLD}).")
    return p.parse_args()


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    args = _parse_args()
    N_MCMC, N_BURNIN, N_SIM, AIC_THRESHOLD = (
        args.n_mcmc, args.n_burnin, args.n_sim, args.aic_threshold)

    run_pipeline(
        run_validation=not args.no_validation,
        use_proxies=not args.no_proxies,
        output_dir=Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
    )
