"""
=============================================================================
STEP 3 / 6  —  RTS smoother (Rauch-Tung-Striebel)
=============================================================================

Reference paper:
    Laine et al. (2014), ACP, Appendix A, equations A9-A11

Code shared with the CH4 pipeline (DLM/dlmpython_ch4/step3_rts_smoother.py):
    the smoother is generic (species-independent).

Original MATLAB (dlm/dlmks.m):
    % Initialization at the last month
    xs(:,n) = x(:,n);
    Ps(:,:,n) = P(:,:,n);

    % Backward pass
    for t = n-1 : -1 : 1
        L         = P(:,:,t) * G' / Pp(:,:,t+1);
        xs(:,t)   = x(:,t)  + L * (xs(:,t+1) - xp(:,t+1));
        Ps(:,:,t) = P(:,:,t) + L * (Ps(:,:,t+1) - Pp(:,:,t+1)) * L';
    end

──────────────────────────────────────────────────────────────────────────────

REMINDER OF THE PROBLEM
───────────────────
After step 2 (forward filter), for each month t we have:
    x_hat(t|t)   : state estimated using data up to t
    P(t|t)       : uncertainty of this estimate

This is not optimal: to estimate the state in 2005, why ignore the
data from 2006 to today?

The RTS smoother fixes this with a single backward pass from T to 1,
producing:
    x_hat_s(t)   : SMOOTHED state using ALL available data
    P_s(t)       : uncertainty (always smaller than P(t|t))

RTS SMOOTHER EQUATIONS (Appendix A)
─────────────────────────────────────────
For t = T-1, T-2, ..., 1  (going back in time):

  Smoother gain:      L(t) = P(t|t) . G^T . [P(t+1|t)]^-1        (Eq. A9)

  Smoothed state:      x_hat_s(t) = x_hat(t|t) + L(t) . delta_x    (Eq. A10)
                       with delta_x = x_hat_s(t+1) - x_hat(t+1|t)

  Smoothed covariance: P_s(t) = P(t|t) + L(t) . delta_P . L(t)^T   (Eq. A11)
                       with delta_P = P_s(t+1) - P(t+1|t)

KEY OUTPUTS
────────────
    x_hat_s(t)[0]  = mu_s(t)  : smoothed ozone level
    x_hat_s(t)[1]  = nu_s(t)  : smoothed slope [%/month]
    P_s(t)[1,1]              : variance of the smoothed slope -> CI on nu_s(t)

The smoothed slope x 120 gives the trend in %/decade.
Its 95% CI = nu_s(t) +/- 1.96 x sqrt(P_s(t)[1,1]) x 120

=============================================================================
"""

import numpy as np
from step1_ssm_matrices import make_G, N_STATE, RHO


def kalman_smoother(kf_result: dict, rho: float = RHO) -> dict:
    """
    RTS smoother (Rauch-Tung-Striebel) — backward pass.

    Takes as input the Kalman filter result (step 2) and produces the
    smoothed states over the whole series.

    Parameters
    ----------
    kf_result : dict
        Output of kalman_filter() — must contain:
            x_filt  (n, N_STATE)          : filtered states x_hat(t|t)
            P_filt  (n, N_STATE, N_STATE) : filtered covariances P(t|t)
            x_pred  (n, N_STATE)          : predicted states x_hat(t+1|t)
            P_pred  (n, N_STATE, N_STATE) : predicted covariances P(t+1|t)

    Returns
    -------
    dict with:
        x_smooth  (n, N_STATE)        : smoothed states x_hat_s(t)
        P_smooth  (n, N_STATE, N_STATE) : smoothed covariances P_s(t)
        L_gains   (n, N_STATE, N_STATE) : smoother gains L(t) — used by the sim. smoother
        level     (n,)       : level mu_s(t)          = x_smooth[:, 0]
        slope     (n,)       : slope nu_s(t) [/month]  = x_smooth[:, 1]
        slope_std (n,)       : slope std dev           = sqrt(P_smooth[:, 1, 1])
        slope_dec (n,)       : slope in %/decade        = slope x 12000
        ic_lower  (n,)       : 95% CI lower bound on slope [%/dec]
        ic_upper  (n,)       : 95% CI upper bound on slope [%/dec]
    """
    x_filt = kf_result["x_filt"]   # (n, N_STATE)
    P_filt = kf_result["P_filt"]   # (n, N_STATE, N_STATE)
    x_pred = kf_result["x_pred"]   # (n, N_STATE) = x_hat(t+1|t) shifted
    P_pred = kf_result["P_pred"]   # (n, N_STATE, N_STATE) = P(t+1|t) shifted
    n      = x_filt.shape[0]
    G      = make_G(rho)  # must match the rho used to produce kf_result

    # ── Allocate output arrays ────────────────────────────────
    x_smooth = np.zeros((n, N_STATE))
    P_smooth = np.zeros((n, N_STATE, N_STATE))
    L_gains  = np.zeros((n, N_STATE, N_STATE))

    # ── Initialization at the last month T ─────────────────────────────
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # ── Backward pass: from T-1 down to 0 ────────────────────────────────
    for t in range(n - 2, -1, -1):

        Pp_next = P_pred[t + 1]     # (N_STATE, N_STATE)

        # ── Eq. A9: Smoother gain L(t) ──────────────────────────────────
        # L = P_filt[t] @ G.T @ inv(Pp_next)
        # <=> L @ Pp_next = P_filt[t] @ G.T
        # <=> Pp_next.T @ L.T = G @ P_filt[t].T
        try:
            L = np.linalg.solve(Pp_next.T, (G @ P_filt[t].T)).T
        except np.linalg.LinAlgError:
            L = P_filt[t] @ G.T @ np.linalg.pinv(Pp_next)

        L_gains[t] = L

        # ── Eq. A10: Smoothed state x_hat_s(t) ─────────────────────────────────
        delta_x     = x_smooth[t + 1] - x_pred[t + 1]
        x_smooth[t] = x_filt[t] + L @ delta_x

        # ── Eq. A11: Smoothed covariance P_s(t) ───────────────────────────
        delta_P     = P_smooth[t + 1] - Pp_next
        P_smooth[t] = P_filt[t] + L @ delta_P @ L.T

        # Force symmetry (numerical round-off errors)
        P_smooth[t] = (P_smooth[t] + P_smooth[t].T) / 2

    # ── Extract physical quantities ───────────────────────────────

    level = x_smooth[:, 0]
    slope = x_smooth[:, 1]

    slope_var = np.maximum(P_smooth[:, 1, 1], 0)
    slope_std = np.sqrt(slope_var)

    slope_dec = slope * 12000
    std_dec   = slope_std * 12000

    ic_lower = slope_dec - 1.96 * std_dec
    ic_upper = slope_dec + 1.96 * std_dec

    return {
        "x_smooth":  x_smooth,    # (n, N_STATE)  — all smoothed states
        "P_smooth":  P_smooth,    # (n, N_STATE, N_STATE) — smoothed covariances
        "L_gains":   L_gains,     # (n, N_STATE, N_STATE) — gains (step 4)
        "level":     level,       # (n,)  mu_s(t)
        "slope":     slope,       # (n,)  nu_s(t) [unit/month]
        "slope_std": slope_std,   # (n,)  std(nu_s(t))
        "slope_dec": slope_dec,   # (n,)  trend [%/dec or unit/dec]
        "ic_lower":  ic_lower,    # (n,)  95% CI lower bound [%/dec]
        "ic_upper":  ic_upper,    # (n,)  95% CI upper bound [%/dec]
    }


# ─── ROLLING TRENDS — Nilsen et al. (2024) ──────────────────────────────

def rolling_trend_from_level(smooth_result: dict,
                             window_yr: int = 20) -> dict:
    """
    Computes rolling trends from the smoothed level mu_s(t).

    Method from Laine (2014) Section 2.5 and Nilsen et al. (2024):
        trend(t) = mu_s(t + W/2) - mu_s(t - W/2)

    Parameters
    ----------
    smooth_result : dict
        Output of kalman_smoother()
    window_yr : int
        Window in years (default: 20 years as in Nilsen 2024, since ozone
        has a slow post-Montreal recovery that requires a long window to
        be resolved)

    Returns
    -------
    dict with:
        trend_dec  (n_valid,) : mean trend over the window [%/dec]
        center_idx (n_valid,) : indices t of the window center
    """
    level  = smooth_result["level"]
    level_std = np.sqrt(np.maximum(smooth_result["P_smooth"][:, 0, 0], 0))
    W = window_yr * 12  # window in months
    half = W // 2

    trends, stds, centers = [], [], []

    for t in range(half, len(level) - half):
        delta_mu = level[t + half] - level[t - half]
        trend_dec = delta_mu * 12000 / W
        se = np.sqrt(level_std[t + half]**2 + level_std[t - half]**2) * 12000 / W

        trends.append(trend_dec)
        stds.append(se)
        centers.append(t)

    return {
        "trend_dec":   np.array(trends),
        "trend_std":   np.array(stds),
        "ic_lower":    np.array(trends) - 1.96 * np.array(stds),
        "ic_upper":    np.array(trends) + 1.96 * np.array(stds),
        "center_idx":  np.array(centers),
    }


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    from step2_kalman_filter import kalman_filter

    np.random.seed(42)
    n = 100
    true_slope = 0.00833
    t_mo = np.arange(n)
    y = (true_slope * t_mo
         + 0.15 * np.sin(2*np.pi*t_mo/12)
         + np.random.normal(0, 0.05, n))

    kf = kalman_filter(y=y, sigma_obs=0.05,
                       sigma_trend=0.0005, sigma_seas=0.01)
    ks = kalman_smoother(kf)

    print(f"True trend     : {true_slope*12000:+.4f} %/dec")
    print(f"Smoothed (end) : nu_s(T) = {ks['slope_dec'][-1]:+.4f} %/dec")
