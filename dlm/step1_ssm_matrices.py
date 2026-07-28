"""
=============================================================================
STEP 1 / 6  --  State-space model matrices
=============================================================================

Reference paper:
    Laine et al. (2014), ACP, doi:10.5194/acp-14-9707-2014
    Appendix A, equations A1-A3

Code shared with the CH4 pipeline (DLM/dlmpython_ch4/step1_ssm_matrices.py):
    the state-space model is generic (species-independent), only the
    number of retained harmonics differs.
    NOTE: Laine et al. (2014)/ozonedemo.m actually uses options.trig=2
    (2 harmonics, 6 states, NO AR(1)) for ozone -- verified directly in
    DLM/dlmmatlab/dlm/examples/ozonedemo.m. N_HARM=2 (7 states with the
    AR(1) term added here) is therefore the configuration retained below,
    matching Laine (2014). A 4-harmonic variant (11 states) was tested
    empirically and rejected (worse DIC on 3 of 4 layers) -- see the
    pipeline report, Section "State-space model".

Original MATLAB (dlm/dlmfit.m, lines ~40-80):
    % Build state transition matrix
    G = blkdiag([1 1; 0 1], trigmat(w1), trigmat(w2), trigmat(w3), trigmat(w4));

    % Build observation matrix
    F = [1 0 1 0 1 0 1 0 1 0];

    % Build noise covariance
    Q = diag([0 s2(1) s2(2) s2(2) s2(2) s2(2) s2(2) s2(2) s2(2) s2(2)]);

──────────────────────────────────────────────────────────────────────────────

STATE-SPACE MODEL
─────────────────────────
The state vector at time t is (default: N_HARM=2, 7 states):

    x(t) = [ mu(t),  nu(t),  gamma_c1(t),  gamma_s1(t),
             gamma_c2(t),  gamma_s2(t),  a(t) ]^T

    mu(t)         : trend level (background ozone)
    nu(t)         : instantaneous slope (trend, %/month)
    gamma_c1, gamma_s1 : annual harmonic     (T = 12 months, omega_1 = 2*pi/12)
    gamma_c2, gamma_s2 : semi-annual harmonic (T = 6 months,  omega_2 = 2*pi/6)
    a(t)          : AR(1) residual, rho estimated by MCMC (see step5)

(Up to 2 more harmonic pairs, T=4 and T=3 months, can be enabled via
N_HARM=3/4 -- see FREQS_ALL below -- but are not part of the retained
configuration.)

TRANSITION EQUATION (Eq. A1):
    x(t) = G . x(t-1) + w(t)      w(t) ~ N(0, Q)

    G = blkdiag( [1 1; 0 1],  trigmat(omega_1),  trigmat(omega_2), ..., rho )

    where trigmat(omega) = [[cos(omega), sin(omega)], [-sin(omega), cos(omega)]]

MEASUREMENT EQUATION (Eq. A2):
    y(t) = F . x(t) + sum_n beta_n * X_n(t) + v(t)      v(t) ~ N(0, R)

    F = [1, 0, 1, 0, ..., 1, 0, 1]  <- mu + cosine parts of the harmonics + AR(1)
    X_n(t) = proxies (QBO, MgII, SAOD, AO, EHF, MEI, EESC, VPSC, T_LS, T_MS, TP...)
    R = sigma_obs^2  (observation variance)

STATE NOISE (Eq. A3):
    Q = diag(0, sigma^2_nu, sigma^2_gamma, ..., sigma^2_gamma, sigma^2_ar)

    sigma^2_mu  = 0   : level mu has no noise of its own (evolves via nu)
    sigma^2_nu        : stochastic slope <- KEY PARAMETER for a time-varying trend
    sigma^2_gamma     : seasonal-cycle variability (shared across all harmonics)
    sigma^2_ar        : AR(1) residual noise

    These parameters are estimated by MCMC in step 5.

=============================================================================
"""

import numpy as np


# ─── MODEL CONSTANTS ──────────────────────────────────────────────────────────

RHO       = 0.95       # initial AR(1) coefficient (re-estimated by MCMC, step5)


# ─── TRANSITION MATRIX G ──────────────────────────────────────────────────────

# Available harmonics (periods in months); N_HARM controls how many are
# actually used. Laine (2014)/ozonedemo.m actually uses N_HARM=2 (trig=2)
# for ozone -- see the note in the module docstring above. Tested here
# empirically against N_HARM=4 (mirroring the analogous 2-vs-4-harmonic
# test done for CH4).
FREQS_ALL = [12.0, 6.0, 4.0, 3.0]
N_HARM    = 2
FREQS     = FREQS_ALL[:N_HARM]

N_STATE   = 2 + 2 * len(FREQS) + 1   # level+slope + harmonics + AR(1)
AR_IDX    = N_STATE - 1               # index of the AR(1) state


def make_G(rho: float = RHO) -> np.ndarray:
    """
    Builds the transition matrix G (N_STATE x N_STATE).
    Last state = AR(1) with coefficient rho.
    """
    G = np.zeros((N_STATE, N_STATE))

    # Trend block (Local Linear Trend)
    G[0, 0] = 1.0;  G[0, 1] = 1.0   # mu receives the slope nu
    G[1, 1] = 1.0

    for k, T in enumerate(FREQS):
        omega = 2 * np.pi / T
        i = 2 + 2*k
        G[i,   i  ] =  np.cos(omega)
        G[i,   i+1] =  np.sin(omega)
        G[i+1, i  ] = -np.sin(omega)
        G[i+1, i+1] =  np.cos(omega)

    # AR(1) state
    G[AR_IDX, AR_IDX] = rho

    return G


# ─── OBSERVATION VECTOR F ─────────────────────────────────────────────────────

def make_F() -> np.ndarray:
    """
    Builds the observation vector F (1 x N_STATE).

    y(t) = mu(t) + sum_k gamma_ck(t) + ar(t)
    """
    F = np.zeros((1, N_STATE))
    F[0, 0] = 1.0   # level mu
    for k in range(len(FREQS)):
        F[0, 2 + 2*k] = 1.0   # cos omega_k
    F[0, AR_IDX] = 1.0   # AR(1)
    return F


# ─── STATE NOISE MATRIX Q ─────────────────────────────────────────────────────

def make_Q(sigma_trend: float, sigma_seas: float,
           sigma_ar: float = 0.0) -> np.ndarray:
    """
    Builds the state noise covariance matrix Q (N_STATE x N_STATE).
    """
    q = np.zeros(N_STATE)
    q[0]   = 0.0
    q[1]   = sigma_trend**2
    q[2:2 + 2*len(FREQS)] = sigma_seas**2
    q[AR_IDX]  = sigma_ar**2
    return np.diag(q)


# ─── OBSERVATION VARIANCE R ───────────────────────────────────────────────────

def make_R(sigma_obs: float) -> np.ndarray:
    """
    Builds the observation variance R (scalar -> 1x1 matrix).

    MATLAB:  R = s^2   (variance of the series y)

    For the ozonesondes, sigma_obs comes from:
      - ECC measurement uncertainty (~5%)
      - residual interannual variability

    In Laine (2014), sigma_obs is estimated by MCMC together with
    sigma_trend and sigma_seas. It can also be fixed to the median of the
    per-point sigma values (anom_df["sigma"]).
    """
    return np.array([[sigma_obs**2]])


# ─── VISUAL CHECK ──────────────────────────────────────────────────────────────

def describe_model(sigma_trend=0.001, sigma_seas=0.01,
                   sigma_obs=0.05, sigma_ar=0.01):
    """Prints the model matrices for inspection."""
    G = make_G()
    F = make_F()
    Q = make_Q(sigma_trend, sigma_seas, sigma_ar)
    R = make_R(sigma_obs)

    print("="*60)
    print("  STATE-SPACE MODEL -- Laine (2014) + AR(1), ozone")
    print("="*60)

    print(f"\n  N_HARM = {N_HARM}, N_STATE = {N_STATE}")
    print(f"\n  G (transition):\n{np.array2string(G, precision=3, suppress_small=True)}")
    print(f"\n  F (observation): {F}")
    print(f"\n  Q (state noise) diagonal: {np.diag(Q)}")
    print(f"\n  R (obs noise)  : sigma_obs = {sigma_obs:.4f}")


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    describe_model()
