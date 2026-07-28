"""
=============================================================================
STEP 5 / 6  --  MCMC (Adaptive Metropolis-Hastings)
=============================================================================

Reference paper:
    Laine et al. (2014), ACP, Section 2.3-2.4, Table 1
    Haario et al. (2001) - Adaptive Metropolis algorithm

Code shared with the CH4 pipeline (DLM/dlmpython_ch4/step5_mcmc.py),
includes the fixes introduced during the critical review of the
validation harness (multi-chain audit for a proper Rhat, see
verif_modele.py):
    - theta_init lets chain starting points be dispersed
    - run_mcmc_multichain() runs several independent chains

PARAMETERS TO ESTIMATE (5)
──────────────────────────
    theta = [ sigma_trend,  sigma_seas,  sigma_obs,  sigma_ar,  rho ]

    sigma_trend : std of the monthly change in slope nu(t)
    sigma_seas  : std of the seasonal-cycle variability
    sigma_obs   : std of the residual observation noise
    sigma_ar    : std of the AR(1) noise
    rho         : AR(1) state persistence -- estimated (see verif_modele.py,
                  Level 1: if innovations stay autocorrelated with rho
                  fixed, promoting it to an MCMC parameter is the fix)

PRIORS (Laine 2014, Table 1 + AR(1))
──────────────────────────────────────
    sigma_trend ~ logN( log(|y_bar| x 5e-5),   sig_prior=1.0 )
    sigma_seas  ~ logN( log(sigma_y x 0.01  ), sig_prior=2.0 )
    sigma_obs   ~ logN( log(sigma_y x 0.30  ), sig_prior=2.0 )
    sigma_ar    ~ logN( log(sigma_y x 0.05  ), sig_prior=1.5 )
    rho         : logit(rho) ~ N( logit(0.9), 1.5 )  -- wide, centered
              around strong persistence, but lets the data decide
              rather than pinning down a fixed value.
"""

import numpy as np
import sys
sys.path.insert(0, ".")
from step2_kalman_filter import kalman_filter


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─── PRIORS (Laine 2014, Table 1) ─────────────────────────────────────────────

def make_prior(y: np.ndarray) -> dict:
    """
    Log-normal priors calibrated on Laine et al. (2014) Table 1,
    plus a prior on logit(rho) for the 11th AR(1) state.
    """
    valid = y[~np.isnan(y)]
    y_mean = np.abs(np.mean(valid))   # |y_bar|
    y_std  = np.std(valid)            # sigma_y

    return {
        "sigma_trend": {
            "mu_log":  np.log(y_mean * 5e-5),
            "sig_log": 1.0,
        },
        "sigma_seas": {
            "mu_log":  np.log(y_std * 0.01),
            "sig_log": 2.0,
        },
        "sigma_obs": {
            "mu_log":  np.log(y_std * 0.30),
            "sig_log": 2.0,
        },
        "sigma_ar": {
            "mu_log":  np.log(y_std * 0.05),
            "sig_log": 1.5,
        },
        "rho": {
            "mu_log":  _logit(0.9),
            "sig_log": 1.5,
        },
    }


PARAM_NAMES = ["sigma_trend", "sigma_seas", "sigma_obs", "sigma_ar", "rho"]


# ─── LOG-PRIOR ────────────────────────────────────────────────────────────────

def log_prior(theta: np.ndarray, prior: dict) -> float:
    """
    Log-prior for theta = [log sigma_trend, log sigma_seas, log sigma_obs,
    log sigma_ar, logit rho].
    """
    lp = 0.0
    for i, name in enumerate(PARAM_NAMES):
        mu  = prior[name]["mu_log"]
        sig = prior[name]["sig_log"]
        lp += -0.5 * ((theta[i] - mu) / sig)**2
    return lp


# ─── LOG-POSTERIOR ────────────────────────────────────────────────────────────

def log_posterior(theta:    np.ndarray,
                  y:          np.ndarray,
                  prior:      dict,
                  proxies:    np.ndarray | None = None,
                  beta:       np.ndarray | None = None) -> float:
    """
    log p(theta | y) = log p(y | theta) + log p(theta)
    theta = [log sigma_trend, log sigma_seas, log sigma_obs, log sigma_ar, logit rho]
    """
    sigma_trend = np.exp(theta[0])
    sigma_seas  = np.exp(theta[1])
    sigma_obs   = np.exp(theta[2])
    sigma_ar    = np.exp(theta[3])
    rho         = np.clip(_sigmoid(theta[4]), 1e-6, 1 - 1e-6)

    # Numerical safeguards
    if sigma_trend < 1e-12 or sigma_seas < 1e-12 or sigma_obs < 1e-12 or sigma_ar < 1e-12:
        return -np.inf
    if sigma_trend > 10 or sigma_seas > 10 or sigma_obs > 10 or sigma_ar > 10:
        return -np.inf

    try:
        kf = kalman_filter(
            y           = y,
            sigma_obs   = sigma_obs,
            sigma_trend = sigma_trend,
            sigma_seas  = sigma_seas,
            sigma_ar    = sigma_ar,
            proxies     = proxies,
            beta        = beta,
            rho         = rho,
        )
        ll = kf["loglik"]
    except Exception:
        return -np.inf

    if not np.isfinite(ll):
        return -np.inf

    lp = log_prior(theta, prior)

    return ll + lp


# ─── ADAPTIVE METROPOLIS MCMC ALGORITHM ──────────────────────────────────────

def run_mcmc(
    y:           np.ndarray,
    prior:       dict,
    n_iter:      int = 5000,
    n_burnin:    int = 2000,
    proxies:     np.ndarray | None = None,
    beta:        np.ndarray | None = None,
    adapt_every: int = 100,
    target_rate: float = 0.234,
    verbose:     bool  = True,
    rng:         np.random.Generator | None = None,
    theta_init:  np.ndarray | None = None,
) -> dict:
    """
    Adaptive Metropolis MCMC to estimate 5 parameters
    (sigma_trend, sigma_seas, sigma_obs, sigma_ar, rho).

    theta_init : explicit starting point (log/logit transformed space).
        If None, starts at the prior mode -- used by run_mcmc_multichain
        to disperse the chains (R-hat diagnostic, Level 2).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # ── Initial OLS estimate of the proxy coefficients ──────────────────
    if proxies is not None and beta is None:
        # valid on BOTH y and the proxies -- a single proxy NaN (outside
        # its coverage period, e.g. TP 1989-2024 on a 1988-2026 series)
        # is enough to make lstsq (SVD) fail if we only filter on y.
        valid = ~np.isnan(y) & ~np.any(np.isnan(proxies), axis=1)
        n_v = valid.sum()
        A     = np.column_stack([np.ones(n_v),
                                 np.arange(len(y))[valid],
                                 proxies[valid]])
        beta, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
        beta  = beta[2:]

    # ── Initialization theta_0 (prior mode, or given dispersed point) ──────
    if theta_init is None:
        theta = np.array([prior[name]["mu_log"] for name in PARAM_NAMES])
    else:
        theta = np.array(theta_init, dtype=float)
    current_lp = log_posterior(theta, y, prior, proxies, beta)

    if not np.isfinite(current_lp):
        raise ValueError("Log-posterior not finite at start -- check the data.")

    # ── Initial proposal covariance ────────────────────────────────
    d = len(PARAM_NAMES)
    C = np.eye(d) * 0.1**2

    # ── Storage ──────────────────────────────────────────────────────────
    n_total   = n_iter + n_burnin
    chains    = np.zeros((n_total, d))
    logposts  = np.zeros(n_total)
    accepted  = 0

    if verbose:
        print(f"\n  MCMC started: {n_total} iterations "
              f"({n_burnin} burn-in + {n_iter} post)")
        print(f"  Initial parameters:")
        print(f"    sigma_trend = {np.exp(theta[0]):.6f}")
        print(f"    sigma_seas  = {np.exp(theta[1]):.4f}")
        print(f"    sigma_obs   = {np.exp(theta[2]):.4f}")
        print(f"    sigma_ar    = {np.exp(theta[3]):.6f}")
        print(f"    rho         = {_sigmoid(theta[4]):.4f}")

    # ── MCMC loop ───────────────────────────────────────────────────────
    for i in range(n_total):

        # 1. AM proposal
        if i > 100:
            C_emp = np.cov(chains[max(0, i-500):i].T)
            if C_emp.ndim == 2 and np.all(np.isfinite(C_emp)):
                C = (2.38**2 / d) * C_emp + 1e-8 * np.eye(d)

        epsilon = rng.multivariate_normal(np.zeros(d), C)
        theta_prop = theta + epsilon

        # 2. Log-posterior of the proposal
        lp_star = log_posterior(theta_prop, y, prior, proxies, beta)

        # 3. Acceptance ratio
        log_alpha = lp_star - current_lp

        # 4. Accept / reject
        if np.log(rng.uniform()) < log_alpha:
            theta      = theta_prop
            current_lp = lp_star
            accepted  += 1

        chains[i]   = theta
        logposts[i] = current_lp

        # 5. Progress printout
        if verbose and (i + 1) % 1000 == 0:
            p0 = np.exp(theta[:4])
            print(f"    iter {i+1:5d} | accept {accepted/(i+1):.1%} | "
                  f"sigma_trend={p0[0]:.5f} | sigma_seas={p0[1]:.4f} | "
                  f"sigma_obs={p0[2]:.4f} | sigma_ar={p0[3]:.6f} | "
                  f"rho={_sigmoid(theta[4]):.4f}")

    final_rate = accepted / n_total

    # ── Post-burn-in ───────────────────────────────────────────────────────
    chains_post   = chains[n_burnin:]
    logposts_post = logposts[n_burnin:]

    # ── Convert to physical parameters ─────────────────────────────────
    params = np.exp(chains_post[:, :4])
    rho_samples = _sigmoid(chains_post[:, 4])

    if verbose:
        print(f"\n  MCMC results (post-burn-in, {n_iter} samples):")
        print(f"  Acceptance rate: {final_rate:.1%}  "
              f"(target: {target_rate:.1%})")
        _print_param_summary("sigma_trend", params[:, 0])
        _print_param_summary("sigma_seas",  params[:, 1])
        _print_param_summary("sigma_obs",   params[:, 2])
        _print_param_summary("sigma_ar",    params[:, 3])
        _print_param_summary("rho",         rho_samples)

    return {
        "chains":       chains_post,
        "sigma_trend":  params[:, 0],
        "sigma_seas":   params[:, 1],
        "sigma_obs":    params[:, 2],
        "sigma_ar":     params[:, 3],
        "rho":          rho_samples,
        "accept_rate":  final_rate,
        "logpost":      logposts_post,
        "n_burnin":     n_burnin,
        "n_iter":       n_iter,
        "beta":         beta,
    }


def run_mcmc_multichain(
    y:           np.ndarray,
    prior:       dict,
    n_chains:    int = 4,
    n_iter:      int = 3000,
    n_burnin:    int = 1000,
    proxies:     np.ndarray | None = None,
    beta:        np.ndarray | None = None,
    seed0:       int = 42,
) -> dict:
    """
    Level 2 convergence audit: several independent chains with dispersed
    starting points (instead of a single chain at the prior mode), to
    allow a proper R-hat (Gelman-Rubin) -- a single chain can never
    detect a hidden mode / convergence to a different optimum depending
    on the start.
    """
    d = len(PARAM_NAMES)
    theta0 = np.array([prior[name]["mu_log"] for name in PARAM_NAMES])
    sig0   = np.array([prior[name]["sig_log"] for name in PARAM_NAMES])

    chains_list = []
    accept_rates = []
    beta_used = beta
    for c in range(n_chains):
        rng_c = np.random.default_rng(seed0 + 1000 * (c + 1))
        jitter = sig0 * 0.8 * (1 if c % 2 == 0 else -1) * (0.5 + c / (2 * n_chains))
        theta_init = theta0 + jitter
        res_c = run_mcmc(y, prior, n_iter=n_iter, n_burnin=n_burnin,
                         proxies=proxies, beta=beta_used,
                         verbose=False, rng=rng_c, theta_init=theta_init)
        if beta_used is None:
            beta_used = res_c["beta"]
        chains_list.append(res_c["chains"])
        accept_rates.append(res_c["accept_rate"])

    pooled = np.concatenate(chains_list, axis=0)
    params = np.exp(pooled[:, :4])
    rho_samples = _sigmoid(pooled[:, 4])

    return {
        "chains":       pooled,
        "chains_list":  chains_list,
        "sigma_trend":  params[:, 0],
        "sigma_seas":   params[:, 1],
        "sigma_obs":    params[:, 2],
        "sigma_ar":     params[:, 3],
        "rho":          rho_samples,
        "accept_rate":  float(np.mean(accept_rates)),
        "n_chains":     n_chains,
        "n_burnin":     n_burnin,
        "n_iter":       n_iter,
        "beta":         beta_used,
    }


def _print_param_summary(name, samples):
    """Prints the mean, median and 95% CI of a parameter."""
    print(f"  {name:<12} : "
          f"median={np.median(samples):.5f}  "
          f"CI95=[{np.percentile(samples,2.5):.5f}, "
          f"{np.percentile(samples,97.5):.5f}]")


# ─── MCMC DIAGNOSTICS ─────────────────────────────────────────────────────────

def mcmc_diagnostics(mcmc_result: dict) -> dict:
    """
    Quick diagnostics on the production chain (single-chain, AR(1)
    approximation for the ESS) -- the rigorous Level 2 check (multi-chain
    Rhat + Geyer ESS) is in verif_modele.py.
    """
    chains = mcmc_result["chains"]   # (N, 5)
    n_params = chains.shape[1]
    display_names = ["log sigma_trend", "log sigma_seas", "log sigma_obs", "log sigma_ar", "logit rho"][:n_params]
    to_physical = [np.exp] * (n_params - 1) + [_sigmoid] if n_params == 5 else [np.exp] * n_params

    print("\n  MCMC DIAGNOSTICS")
    print("  " + "-"*50)

    rate = mcmc_result["accept_rate"]
    flag = "OK" if 0.15 < rate < 0.40 else "WARNING"
    print(f"  Acceptance rate: {rate:.1%}  [{flag}]")
    print(f"  (ideal range 15%-40% for {n_params} parameters)")

    print(f"\n  Lag-1 autocorrelation (mixing time):")
    ess_list = []
    for i, name in enumerate(display_names):
        chain_i = chains[:, i]
        r1 = np.corrcoef(chain_i[:-1], chain_i[1:])[0, 1]
        n  = len(chain_i)
        ess = n * (1 - r1) / (1 + r1)
        ess_list.append(ess)
        flag = "OK" if ess > 100 else "CHAIN TOO SHORT"
        print(f"    {name:<14} : r1={r1:.3f}  ESS~{ess:.0f}  [{flag}]")

    n2 = len(chains) // 2
    print(f"\n  Stationarity (1st vs 2nd half):")
    for i, name in enumerate(display_names):
        transform = to_physical[i]
        m1  = transform(chains[:n2, i]).mean()
        m2  = transform(chains[n2:, i]).mean()
        rng_val = max(abs(m1), abs(m2), 1e-12)
        rel = abs(m1 - m2) / rng_val
        flag = "OK" if rel < 0.1 else "NON-STATIONARY"
        print(f"    {name:<14} : {m1:.5f} vs {m2:.5f}  (diff {rel:.1%})  [{flag}]")

    return {"ess": ess_list, "accept_rate": rate}


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    np.random.seed(42)
    n = 300
    TRUE_SIGMA_TREND, TRUE_SIGMA_SEAS = 0.0003, 0.008
    TRUE_SIGMA_OBS, TRUE_SIGMA_AR, TRUE_RHO = 0.04, 0.005, 0.9

    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = TRUE_RHO * ar[i - 1] + np.random.normal(0, TRUE_SIGMA_AR)

    t_mo  = np.arange(n)
    y_obs = (0.00833 * t_mo + 0.12 * np.sin(2*np.pi*t_mo/12)
             + ar + np.random.normal(0, TRUE_SIGMA_OBS, n))

    prior = make_prior(y_obs)
    result = run_mcmc(y_obs, prior, n_iter=1500, n_burnin=500, verbose=True)
    mcmc_diagnostics(result)
