"""
BatchBayesian_nmr_corezzi.py

Corezzi Eq. 8 ODE fit to NMR CPMG alpha(t) data.
r = 2.0 fixed by stoichiometry.
B and a0 fixed per (sample, temp) from cpmg_fit_results/t2_alpha_fits.csv.

Corezzi diffusion-corrected rate:
  k_eff_i = k_ci / (1 + (k_ci / k0) * exp(xi * B * alpha / (a0 - alpha)))
  dα/dt = (k_eff1 + k_eff2 * α^m) * (1-α)^(n/2) * (r-α)^(n/2)

Usage:
  python BatchBayesian_nmr_corezzi.py                  # LS all datasets
  python BatchBayesian_nmr_corezzi.py EDA 25C          # LS single dataset
  python BatchBayesian_nmr_corezzi.py --mcmc           # LS + MCMC all
  python BatchBayesian_nmr_corezzi.py --mcmc EDA 25C   # LS + MCMC single
"""

import os
import argparse
import pickle
import multiprocessing
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
from numpy import gradient
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee
import corner

from mcmc_config import burnin, nwalkers, nsteps, overlay_n

# ── Constants ─────────────────────────────────────────────────────────────────
R           = 2.0          # stoichiometric ratio, fixed for all samples
DATA_CSV    = "cpmg_fit_results/all_samples.csv"
T2A_CSV     = "cpmg_fit_results/t2_alpha_fits.csv"
KM_CSV      = "fit_results_nmr/km_results.csv"   # for RSS comparison (optional)
OUT_DIR     = "fit_results_nmr"
NPZ_DIR     = "mcmc_samples_nmr"
PLOT_DIR    = "fit_plots_nmr"
PARAM_NAMES = ["log_kc1", "log_kc2", "m", "n", "xi", "log_k0", "log_sigma"]

# ── Load fixed B, a0 table ────────────────────────────────────────────────────
_t2a = pd.read_csv(T2A_CSV)
_t2a_index = {(row["sample"], row["temp"]): row for _, row in _t2a.iterrows()}


def get_B_a0(sample, temp):
    """Return (B, a0) for (sample, temp), or None if not in table."""
    row = _t2a_index.get((sample, temp))
    if row is None:
        return None
    return float(row["B"]), float(row["a0"])


# ── ODE model ─────────────────────────────────────────────────────────────────
def ode_rhs_corezzi(t, a, log_kc1, log_kc2, m, n, xi, log_k0, B, a0, eps=1e-10):
    kc1 = 10 ** log_kc1
    kc2 = 10 ** log_kc2
    k0  = 10 ** log_k0
    # alpha ∈ (0,1); clip to avoid NaN from fractional powers of negative (1-a)
    a   = np.clip(a, eps, min(1.0, a0) - eps)
    mob   = np.exp(xi * B * a / (a0 - a))
    keff1 = kc1 / (1 + (kc1 / k0) * mob)
    keff2 = kc2 / (1 + (kc2 / k0) * mob)
    return (keff1 + keff2 * a**m) * (1 - a)**(n / 2) * (R - a)**(n / 2)


def solve_model(log_kc1, log_kc2, m, n, xi, log_k0, t_data, a_data, B, a0, eps=1e-10):
    try:
        a_init = np.clip(a_data[0], eps, min(1.0, a0) - eps)
        sol    = solve_ivp(
            ode_rhs_corezzi,
            [t_data[0], t_data[-1]],
            [a_init],
            t_eval=t_data,
            args=(log_kc1, log_kc2, m, n, xi, log_k0, B, a0),
            method="LSODA",
            rtol=1e-8, atol=1e-10,
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full_like(t_data, np.nan)
        return np.clip(sol.y[0], 1e-8, min(1.0, a0) - 1e-8)
    except Exception:
        return np.full_like(t_data, np.nan)


# ── Bayesian inference ─────────────────────────────────────────────────────────
def log_prior(params):
    log_kc1, log_kc2, m, n, xi, log_k0, log_sigma = params
    if not (-10 < log_kc1 < 0 and -10 < log_kc2 < 0 and
             0 < m < 5 and 0 < n < 5 and
             0 <= xi < 10 and
            -10 < log_k0 < 0 and
            -12 < log_sigma < 0):
        return -np.inf
    return 0.0


def log_likelihood(params, t_data, a_data, B, a0):
    log_kc1, log_kc2, m, n, xi, log_k0, log_sigma = params
    a_fit = solve_model(log_kc1, log_kc2, m, n, xi, log_k0, t_data, a_data, B, a0)
    if not np.all(np.isfinite(a_fit)):
        return -np.inf
    if np.any(a_fit <= 0) or np.any(a_fit >= R):
        return -np.inf
    sigma    = 10 ** log_sigma
    residual = (a_data - a_fit) / sigma
    return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2))


def log_posterior(params, t_data, a_data, B, a0):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, t_data, a_data, B, a0)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ── MCMC ──────────────────────────────────────────────────────────────────────
def run_mcmc(start_params, t_data, a_data, B, a0, scale_params=None):
    ndim = len(start_params)
    if scale_params is None:
        scale_params = np.array([0.2] * ndim)

    p0 = np.array([
        start_params + scale_params * np.random.randn(ndim)
        for _ in range(nwalkers)
    ])

    test_logps = np.array([log_posterior(p, t_data, a_data, B, a0) for p in p0])
    n_finite   = np.sum(np.isfinite(test_logps))
    if n_finite == 0:
        print("  All initial log-posteriors are -inf. Aborting MCMC.")
        return None
    if n_finite < nwalkers:
        print(f"  Warning: only {n_finite}/{nwalkers} initial positions are finite.")

    with multiprocessing.Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=(t_data, a_data, B, a0), pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True, store=True)

    print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    chain   = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, flat=True)
    return samples, chain, sampler


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(samples, t_data, a_data, sample, temp, B, a0):
    os.makedirs(PLOT_DIR, exist_ok=True)
    label = f"NMR_{sample}_{temp}_corezzi"

    fig = corner.corner(samples, labels=PARAM_NAMES)
    fig.savefig(f"{PLOT_DIR}/{label}_corner.png")
    plt.close(fig)

    idx         = np.random.choice(len(samples), size=min(overlay_n, len(samples)), replace=False)
    alpha_preds = np.array([
        solve_model(*samples[i][:-1], t_data, a_data, B, a0) for i in idx
    ])
    alpha_preds = alpha_preds[np.all(np.isfinite(alpha_preds), axis=1)]
    mean_alpha  = np.nanmean(alpha_preds, axis=0)
    lower_alpha = np.nanpercentile(alpha_preds, 2.5,  axis=0)
    upper_alpha = np.nanpercentile(alpha_preds, 97.5, axis=0)

    dadt_preds = np.array([gradient(a, t_data) for a in alpha_preds])
    mean_dadt  = np.nanmean(dadt_preds, axis=0)
    lower_dadt = np.nanpercentile(dadt_preds, 2.5,  axis=0)
    upper_dadt = np.nanpercentile(dadt_preds, 97.5, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(t_data, a_data, "o", label="Data")
    axes[0].plot(t_data, mean_alpha, label="Median fit")
    axes[0].fill_between(t_data, lower_alpha, upper_alpha, alpha=0.3, label="95% CI")
    axes[0].set(title=f"{label}: α(t)", xlabel="Time (s)", ylabel="α")
    axes[0].legend()

    axes[1].plot(mean_alpha, mean_dadt, label="Median")
    axes[1].fill_between(mean_alpha, lower_dadt, upper_dadt, alpha=0.3, label="95% CI")
    axes[1].set(title=f"{label}: dα/dt vs α", xlabel="α", ylabel="dα/dt")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{label}_combined.png")
    plt.close(fig)


# ── Per-dataset fit ────────────────────────────────────────────────────────────
def load_dataset(sample, temp):
    df   = pd.read_csv(DATA_CSV)
    mask = (df["sample"] == sample) & (df["temp"] == temp) & (~df["dropped"]) & df["alpha"].notna()
    grp  = df[mask].sort_values("elapsed_min")
    if grp.empty:
        return None
    t = grp["elapsed_min"].values.astype(float) * 60  # convert min → s
    a = grp["alpha"].values.astype(float)
    t = t - t[0]
    a = np.clip(a, 1e-8, R - 1e-8)
    return t, a


def fit_ls(t_data, a_data, B, a0):
    """Nelder-Mead LS fit. Returns (params, rss, converged)."""
    # Start xi small (near-KM), k0 large (weak diffusion limit)
    start = np.array([-5.0, -2.0, 0.5, 1.4, 0.5, -2.0, -4.0])
    opt   = minimize(
        lambda p: -log_posterior(p, t_data, a_data, B, a0),
        start,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6},
    )
    params = opt.x
    a_fit  = solve_model(*params[:-1], t_data, a_data, B, a0)
    rss    = float(np.sum((a_data - a_fit)**2)) if np.all(np.isfinite(a_fit)) else np.nan
    return params, rss, opt.success


def process_single(sample, temp, do_mcmc=False, km_rss=None):
    label = f"{sample} {temp}"
    print(f"Fitting NMR {label} (Corezzi, r={R})")

    Ba0 = get_B_a0(sample, temp)
    if Ba0 is None:
        print(f"  No B/a0 for {label} in {T2A_CSV}. Skipping.")
        return None
    B, a0 = Ba0
    print(f"  Fixed: B={B:.4f}, a0={a0:.4f}")

    data = load_dataset(sample, temp)
    if data is None:
        print(f"  No alpha data for {label}. Skipping.")
        return None
    t_data, a_data = data

    params, rss, converged = fit_ls(t_data, a_data, B, a0)
    log_kc1, log_kc2, m, n, xi, log_k0, log_sigma = params
    kc1, kc2, k0 = 10**log_kc1, 10**log_kc2, 10**log_k0

    km_note = f"  KM RSS: {km_rss:.4e} | " if km_rss is not None else ""
    print(f"  LS: kc1={kc1:.2e}, kc2={kc2:.2e}, m={m:.3f}, n={n:.3f}, "
          f"xi={xi:.3f}, k0={k0:.2e}, {km_note}Corezzi RSS={rss:.4e}, converged={converged}")

    row = {
        "sample": sample, "temp": temp,
        "kc1": kc1, "kc2": kc2, "m": m, "n": n,
        "xi": xi, "k0": k0,
        "B_fixed": B, "a0_fixed": a0,
        "rss": rss, "converged": converged,
    }

    if not do_mcmc:
        return row

    scale  = np.array([0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1])
    result = run_mcmc(params, t_data, a_data, B, a0, scale)
    if result is None:
        return row
    samples, chain, sampler = result

    os.makedirs(NPZ_DIR, exist_ok=True)
    tag = f"{sample}_{temp}_corezzi"
    np.savez(
        f"{NPZ_DIR}/{tag}_fitdata.npz",
        samples=samples, chain=chain,
        t_data=t_data, a_data=a_data,
        log_prob=sampler.get_log_prob(),
        B=B, a0=a0,
    )
    with open(f"{NPZ_DIR}/{tag}_sampler.pkl", "wb") as f:
        pickle.dump(sampler, f)

    for i, name in enumerate(PARAM_NAMES):
        vals = 10**samples[:, i] if "log_" in name else samples[:, i]
        row[f"{name}_median"] = np.median(vals)
        row[f"{name}_p2_5"]   = np.percentile(vals, 2.5)
        row[f"{name}_p97_5"]  = np.percentile(vals, 97.5)

    plot_results(samples, t_data, a_data, sample, temp, B, a0)
    return row


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("sample", nargs="?", choices=["EDA", "DAP", "DAB", "DAP2"])
    parser.add_argument("temp",   nargs="?", help="Temperature string, e.g. 25C")
    parser.add_argument("--mcmc", action="store_true", help="Run MCMC after LS fit")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load KM RSS for comparison if available
    km_rss_map = {}
    if os.path.exists(KM_CSV):
        km_df = pd.read_csv(KM_CSV)
        for _, row in km_df.iterrows():
            km_rss_map[(row["sample"], row["temp"])] = row.get("rss", np.nan)

    if args.sample and args.temp:
        km_rss = km_rss_map.get((args.sample, args.temp))
        result = process_single(args.sample, args.temp, do_mcmc=args.mcmc, km_rss=km_rss)
        if result:
            pd.DataFrame([result]).to_csv(
                f"{OUT_DIR}/corezzi_{args.sample}_{args.temp}.csv", index=False
            )
    else:
        df    = pd.read_csv(DATA_CSV)
        tasks = df[["sample", "temp"]].drop_duplicates().values.tolist()

        all_results = []
        for sample, temp in tasks:
            km_rss = km_rss_map.get((sample, temp))
            row    = process_single(sample, temp, do_mcmc=args.mcmc, km_rss=km_rss)
            if row:
                all_results.append(row)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{OUT_DIR}/corezzi_results.csv", index=False)
        print(f"\nResults saved to {OUT_DIR}/corezzi_results.csv")
        print(results_df[["sample", "temp", "kc1", "kc2", "m", "n", "xi", "k0", "rss", "converged"]].to_string(index=False))
