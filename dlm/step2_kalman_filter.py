"""
=============================================================================
STEP 2 / 6  —  Kalman filter (forward pass)
=============================================================================

Reference paper:
    Laine et al. (2014), ACP
    Appendix A, equations A4-A8

Code shared with the CH4 pipeline (DLM/dlmpython_ch4/step2_kalman_filter.py):
    the filter is generic (species-independent).

──────────────────────────────────────────────────────────────────────────────

KALMAN FILTER — PRINCIPLE
─────────────────────────

The Kalman filter solves the problem:
    "Given y(1), ..., y(t), what is the best estimate of x(t)?"

It alternates two steps at each time step:

    ┌──────────────────────────────────────────────────────────────┐
    │ PREDICTION (before observing y(t))                          │
    │                                                              │
    │  x_hat(t|t-1) = G . x_hat(t-1|t-1)          (Eq. A4)         │
    │  P(t|t-1) = G . P(t-1|t-1) . G^T + Q        (Eq. A5)         │
    └──────────────────────────────────────────────────────────────┘
                          ↓  observe y(t)
    ┌──────────────────────────────────────────────────────────────┐
    │ UPDATE (after observing y(t))                                │
    │                                                              │
    │  v(t) = y(t) - F.x_hat(t|t-1) - sum_n beta_n*X_n(t) (Eq. A6) │
    │  S(t) = F.P(t|t-1).F^T + R                          (Eq. A7) │
    │  K(t) = P(t|t-1).F^T . S(t)^-1              (Eq. A8 — gain)  │
    │                                                              │
    │  x_hat(t|t) = x_hat(t|t-1) + K(t).v(t)                       │
    │  P(t|t)  = (I - K(t).F) . P(t|t-1)                            │
    └──────────────────────────────────────────────────────────────┘

    x_hat(t|t) : estimate of x(t) given all observations up to t
    P(t|t)     : estimation-error covariance matrix
    v(t)       : innovation (prediction error of y)
    S(t)       : innovation variance
    K(t)       : Kalman gain — weighs the innovation vs. the prediction

MAIN OUTPUT:
    log-likelihood  L = -1/2 sum [ log|S(t)| + v(t)^2/S(t) ]
    -> used by MCMC to evaluate the parameters (sigma_trend, sigma_seas, sigma_obs)

=============================================================================
"""

import numpy as np
from step1_ssm_matrices import make_G, make_F, make_Q, make_R, N_STATE, RHO


def kalman_filter(
    y:          np.ndarray,       # (n,)   observations (relative ozone anomaly)
    sigma_obs:  float,            # sigma_obs  (observation noise)
    sigma_trend: float,           # sigma_trend (slope noise — key parameter)
    sigma_seas:  float,           # sigma_seas  (seasonal noise)
    sigma_ar:   float = 0.0,     # sigma_ar   (AR(1) noise)
    proxies:    np.ndarray | None = None,  # (n, p) geophysical proxies
    beta:       np.ndarray | None = None,  # (p,)   proxy coefficients
    rho:        float = RHO,     # rho      (AR(1) persistence -- estimated by MCMC, see step5)
) -> dict:
    """
    Forward Kalman filter over the series y.

    Parameters
    ----------
    y           : time series (n points, NaN allowed for gaps)
    sigma_obs   : observation standard deviation
    sigma_trend : standard deviation of the monthly change in slope <- KEY PARAMETER
    sigma_seas  : standard deviation of seasonal variability
    sigma_ar    : standard deviation of the AR(1) noise
    proxies     : proxy matrix  (n x p), optional
    beta        : proxy regression coefficients  (p,), optional
    rho         : AR(1) state persistence coefficient

    Returns
    -------
    dict with:
        loglik    : log-likelihood (used by MCMC)
        x_filt    : filtered states x_hat(t|t)          (n x N_STATE)
        P_filt    : filtered covariances P(t|t)         (n x N_STATE x N_STATE)
        x_pred    : predicted states x_hat(t|t-1)        (n x N_STATE)
        P_pred    : predicted covariances P(t|t-1)       (n x N_STATE x N_STATE)
        v         : innovations v(t)                    (n,)
        S         : innovation variances S(t)            (n,)
    """
    n  = len(y)
    G  = make_G(rho)
    F  = make_F()
    Q  = make_Q(sigma_trend, sigma_seas, sigma_ar)

    R  = make_R(sigma_obs)  # (1, 1)

    # Proxies
    if proxies is None:
        proxies = np.zeros((n, 1))
        beta    = np.zeros(1)
    if beta is None:
        beta = np.zeros(proxies.shape[1])
    beta = np.asarray(beta).reshape(-1)

    # ── Diffuse initialization ──────────────────────────────────────────────
    # x_hat(1|0) = 0  (default centered value if working on anomalies)
    # P(1|0)  = kappa*I  with kappa large  (high initial uncertainty)
    # Laine uses kappa = 1e6 * var(y)
    # Convention (Laine's dlmsmo.m): x0/C0 are directly the prediction at
    # the first step, with no propagation through G -- there is no "t=0"
    # before the first observation.
    kappa = 1e6 * np.nanvar(y)
    x_p   = np.zeros(N_STATE)
    P_p   = kappa * np.eye(N_STATE)

    # Storage
    x_pred = np.zeros((n, N_STATE))
    P_pred = np.zeros((n, N_STATE, N_STATE))
    x_filt = np.zeros((n, N_STATE))
    P_filt = np.zeros((n, N_STATE, N_STATE))
    v_all  = np.full(n, np.nan)
    S_all  = np.full(n, np.nan)
    loglik = 0.0

    for t in range(n):

        x_pred[t] = x_p
        P_pred[t] = P_p

        # ── UPDATE (only if y(t) is not NaN) ────────────────────────────────
        if np.isnan(y[t]):
            # No observation -> keep the prediction
            x_f, P_f = x_p, P_p
        else:
            # Contribution of the proxies to the predicted observation
            proxy_contrib = proxies[t] @ beta  # scalar

            # Eq. A6:  v(t) = y(t) - F . x_hat(t|t-1) - sum_n beta_n*X_n(t)
            v = y[t] - (F @ x_p)[0] - proxy_contrib

            # Eq. A7:  S(t) = F . P(t|t-1) . F^T + R
            S = (F @ P_p @ F.T + R)[0, 0]

            # Eq. A8:  K(t) = P(t|t-1) . F^T / S(t)
            K = (P_p @ F.T) / S   # (N_STATE, 1)

            # State update
            x_f = x_p + K[:, 0] * v

            # Covariance update (Joseph form for numerical stability)
            # P = (I - K.F) . P_p . (I - K.F)^T + K . R . K^T
            IKF = np.eye(N_STATE) - K @ F
            P_f = IKF @ P_p @ IKF.T + K * R[0, 0] @ K.T

            v_all[t] = v
            S_all[t] = S

            # Log-likelihood (implicit Eq. — Kalman filter = MLE)
            # L += -1/2 [log(2*pi) + log(S) + v^2/S]
            loglik += -0.5 * (np.log(2*np.pi) + np.log(S) + v**2 / S)

        x_filt[t] = x_f
        P_filt[t] = P_f

        # ── PREDICTION for the next step ─────────────────────────────────────
        # Eq. A4:  x_hat(t+1|t) = G . x_hat(t|t)
        # Eq. A5:  P(t+1|t) = G . P(t|t) . G^T + Q
        if t < n - 1:
            x_p = G @ x_f
            P_p = G @ P_f @ G.T + Q

    return {
        "loglik": loglik,
        "x_filt": x_filt,    # (n, N_STATE)
        "P_filt": P_filt,    # (n, N_STATE, N_STATE)
        "x_pred": x_pred,    # (n, N_STATE)
        "P_pred": P_pred,    # (n, N_STATE, N_STATE)
        "v":      v_all,     # (n,)  innovations
        "S":      S_all,     # (n,)  innovation variances
    }


# ─── CHECK ─────────────────────────────────────────────────────────────────────

def _demo_filter():
    """Quick test on a synthetic series."""
    np.random.seed(42)
    n = 100

    # Synthetic series: trend + season + noise
    t_mo = np.arange(n)
    y = (0.01*t_mo                          # trend +1%/10yr
         + 0.15*np.sin(2*np.pi*t_mo/12)    # seasonality
         + np.random.normal(0, 0.05, n))   # noise

    # Filter with parameters close to Laine (2014) Table 1
    result = kalman_filter(
        y           = y,
        sigma_obs   = 0.05,
        sigma_trend = 0.0005,   # very small -> near-linear trend
        sigma_seas  = 0.01,
    )

    print(f"Log-likelihood    : {result['loglik']:.2f}")
    print(f"Final level mu    : {result['x_filt'][-1, 0]:.4f}")
    print(f"Final slope nu    : {result['x_filt'][-1, 1]*120:.4f} %/dec")
    print(f"Mean innovation   : {np.nanmean(result['v']):.4f}")
    print(f"Innovation std    : {np.nanstd(result['v']):.4f}")


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    _demo_filter()
