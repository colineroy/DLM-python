"""
=============================================================================
STEP 4 / 6  —  Simulation Smoother (Carter-Kohn, 1994)
=============================================================================

Reference paper:
    Laine et al. (2014), ACP, doi:10.5194/acp-14-9707-2014
    Section 2.4 + Appendix A, Eq. A12

Code shared with the CH4 pipeline (DLM/dlmpython_ch4/step4_simulation_smoother.py):
    the algorithm is generic (species-independent).

Original MATLAB (dlm/dlmsmo.m + dlmsmosam.m):
    % Draw at the last month
    xs(:,n) = mvnrnd(xss(:,n), Pss(:,:,n))';

    % Backward pass
    for t = n-1:-1:1
        mn = xss(:,t) + L(:,:,t) * (xs(:,t+1) - xss(:,t+1));
        Vn = Pss(:,:,t) - L(:,:,t) * Pss(:,:,t+1) * L(:,:,t)';
        xs(:,t) = mvnrnd(mn, Vn)';
    end

──────────────────────────────────────────────────────────────────────────────

WHY THIS IS NEEDED
───────────────────────────
Step 3 (RTS smoother) gives a single trajectory: the MEAN. But it
implicitly assumes the uncertainties are Gaussian and symmetric around
that mean.

BUT step 5 (MCMC) needs the log-likelihood marginalized over the
states — and that requires drawing whole sampled trajectories, not
just the mean.

The Carter-Kohn algorithm draws complete trajectories
{x_t}_{t=1}^{T} from the posterior distribution p(x_{1:T} | y_{1:T}, theta).

ALGORITHM (Carter-Kohn backward sampler):
─────────────────────────────────────────────

1. At the last month T:
   Draw x*(T) ~ N( x_hat_s(T) ,  P_s(T) )

2. For t = T-1, T-2, ..., 0:

   Conditional mean:
       m_t = x_hat_s(t) + L(t) . [x*(t+1) - x_hat_s(t+1)]     (Eq. A12)

   Conditional covariance:
       V_t = P_s(t) - L(t) . P_s(t+1) . L(t)^T

   Draw x*(t) ~ N( m_t ,  V_t )

3. Repeat M times (e.g. M=200) to obtain an ensemble of samples.

=============================================================================
"""

import numpy as np
from step1_ssm_matrices import N_STATE, FREQS


def simulation_smoother(
    smooth_result: dict,
    n_samples:     int = 200,
    rng:           np.random.Generator | None = None,
) -> dict:
    """
    Simulation smoother (Carter-Kohn 1994) — draws M state trajectories.

    Parameters
    ----------
    smooth_result : dict
        Output of kalman_smoother()
    n_samples : int
        Number of trajectories to draw (default: 200, as in Laine 2014)
    rng : np.random.Generator, optional

    Returns
    -------
    dict with:
        samples      (M, n, N_STATE) : M complete state trajectories
        level_samp   (M, n)    : levels mu*(t) = samples[:, :, 0]
        slope_samp   (M, n)    : slopes  nu*(t) = samples[:, :, 1] x 12000 [%/dec]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x_smooth = smooth_result["x_smooth"]   # (n, N_STATE)
    P_smooth = smooth_result["P_smooth"]   # (n, N_STATE, N_STATE)
    L_gains  = smooth_result["L_gains"]    # (n, N_STATE, N_STATE)
    n        = x_smooth.shape[0]

    samples = np.zeros((n_samples, n, N_STATE))

    print(f"  Simulation smoother: {n_samples} trajectories x {n} months...")

    for s in range(n_samples):

        x_curr = _safe_mvnormal(rng, x_smooth[-1], P_smooth[-1])
        samples[s, -1] = x_curr

        for t in range(n - 2, -1, -1):

            correction = L_gains[t] @ (x_curr - x_smooth[t + 1])
            m_t        = x_smooth[t] + correction

            V_t = (P_smooth[t]
                   - L_gains[t] @ P_smooth[t + 1] @ L_gains[t].T)

            x_curr     = _safe_mvnormal(rng, m_t, V_t)
            samples[s, t] = x_curr

        if (s + 1) % 50 == 0:
            print(f"    {s+1}/{n_samples} trajectories drawn")

    level_samp = samples[:, :, 0]           # (M, n)
    slope_samp = samples[:, :, 1] * 12000   # (M, n)  [%/dec]
    harmonic_samp = samples[:, :, 2:2+2*len(FREQS)]  # (M, n, 2*N_HARM)

    print(f"  [OK] {n_samples} trajectories generated")

    return {
        "samples":       samples,        # (M, n, N_STATE) — complete trajectories
        "level_samp":    level_samp,     # (M, n)    — levels mu*(t)
        "slope_samp":    slope_samp,     # (M, n)    — slopes  nu*(t) [%/dec]
        "harmonic_samp": harmonic_samp,  # (M, n, 2*N_HARM) — gamma_c1..gamma_s_N
    }


def _safe_mvnormal(rng, mean, cov):
    """
    Robust multivariate Gaussian draw.

    np.random.multivariate_normal fails if cov is not positive definite
    (which can happen due to numerical round-off errors).
    """
    eps = 1e-10
    cov_reg = cov + eps * np.eye(len(mean))
    cov_reg = (cov_reg + cov_reg.T) / 2

    try:
        return rng.multivariate_normal(mean, cov_reg)
    except np.linalg.LinAlgError:
        U, s_vals, Vt = np.linalg.svd(cov_reg)
        s_vals = np.maximum(s_vals, 1e-12)
        cov_fixed = U @ np.diag(s_vals) @ Vt
        return rng.multivariate_normal(mean, cov_fixed)


# ─── MCMC ROLLING TRENDS ─────────────────────────────────────────────────────

def rolling_trends_mcmc(sim_result: dict,
                        window_yr:  int = 20) -> dict:
    """
    Computes rolling trends over the M samples.

    trend*(t) = mu*(t + W/2) - mu*(t - W/2)   [for each sample]
    """
    level_samp = sim_result["level_samp"]   # (M, n)
    M, n       = level_samp.shape
    W          = window_yr * 12
    half       = W // 2

    n_valid = n - W
    if n_valid <= 0:
        raise ValueError(f"Series too short for a {window_yr}-year window.")

    future_levels = level_samp[:, W:]
    past_levels   = level_samp[:, :n - W]

    delta   = future_levels - past_levels
    trends  = delta / W * 12000

    trend_mean = np.mean(trends, axis=0)
    trend_p025 = np.percentile(trends, 2.5,  axis=0)
    trend_p975 = np.percentile(trends, 97.5, axis=0)
    trend_std  = np.std(trends, axis=0)

    center_idx = np.arange(half, n - half)

    return {
        "trend_mean": trend_mean,
        "trend_p025": trend_p025,
        "trend_p975": trend_p975,
        "trend_std":  trend_std,
        "center_idx": center_idx,
    }


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    from step2_kalman_filter import kalman_filter
    from step3_rts_smoother  import kalman_smoother

    np.random.seed(0)
    n = 200
    true_trend = +1.5
    t_mo       = np.arange(n)
    true_level = true_trend / 120 * t_mo
    y = (true_level
         + 0.12 * np.sin(2*np.pi*t_mo/12)
         + np.random.normal(0, 0.04, n))

    kf  = kalman_filter(y=y, sigma_obs=0.04,
                        sigma_trend=0.0003, sigma_seas=0.008)
    ks  = kalman_smoother(kf)
    sim = simulation_smoother(ks, n_samples=300)

    mean_samp  = sim["level_samp"].mean(axis=0)
    diff_mean  = np.abs(mean_samp - ks["level"]).mean()
    print(f"Mean gap, sampled vs RTS-smoothed: {diff_mean:.6f} (should be small)")
