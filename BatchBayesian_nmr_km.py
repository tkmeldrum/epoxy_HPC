"""
BatchBayesian_nmr_km.py

Kamal-Malkin ODE fit to NMR CPMG alpha(t) data.
r = 2.0 fixed by stoichiometry.

Usage:
  python BatchBayesian_nmr_km.py                  # LS all datasets
  python BatchBayesian_nmr_km.py EDA 25C          # LS single dataset
  python BatchBayesian_nmr_km.py --mcmc           # LS + MCMC all
  python BatchBayesian_nmr_km.py --mcmc EDA 25C   # LS + MCMC single
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
R = 2.0          # stoichiometric ratio, fixed for all samples
DATA_CSV = "cpmg_fit_results/all_samples.csv"
OUT_DIR  = "fit_results_nmr"
NPZ_DIR  = "mcmc_samples_nmr"
PLOT_DIR = "fit_plots_nmr"
PARAM_NAMES = ["log_k1", "log_k2", "m", "n", "log_sigma"]

# ── ODE model ─────────────────────────────────────────────────────────────────
def ode_rhs(t, a, log_k1, log_k2, m, n, eps=1e-10):
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    # alpha is degree of cure ∈ (0,1); clip to avoid NaN from fractional powers
    # of negative (1-a) when ODE overshoots, regardless of r
    a  = np.clip(a, eps, 1 - eps)
    return (k1 + k2 * a**m) * (1 - a)**(n / 2) * (R - a)**(n / 2)


def solve_model(log_k1, log_k2, m, n, t_data, a_data, eps=1e-10):
    try:
        a0  = np.clip(a_data[0], eps, R - eps)
        sol = solve_ivp(
            ode_rhs,
            [t_data[0], t_data[-1]],
            [a0],
            t_eval=t_data,
            args=(log_k1, log_k2, m, n),
            method="LSODA",
            rtol=1e-8, atol=1e-10,
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full_like(t_data, np.nan)
        return np.clip(sol.y[0], 1e-8, R - 1e-8)
    except Exception:
        return np.full_like(t_data, np.nan)


# ── Bayesian inference ─────────────────────────────────────────────────────────
def log_prior(params):
    log_k1, log_k2, m, n, log_sigma = params
    if not (-10 < log_k1 < 0 and -10 < log_k2 < 0 and
             0 < m < 5 and 0 < n < 5 and -12 < log_sigma < 0):
        return -np.inf
    return 0.0


def log_likelihood(params, t_data, a_data):
    log_k1, log_k2, m, n, log_sigma = params
    a_fit = solve_model(log_k1, log_k2, m, n, t_data, a_data)
    if not np.all(np.isfinite(a_fit)):
        return -np.inf
    if np.any(a_fit <= 0) or np.any(a_fit >= R):
        return -np.inf
    sigma    = 10 ** log_sigma
    residual = (a_data - a_fit) / sigma
    return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2))


def log_posterior(params, t_data, a_data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, t_data, a_data)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ── MCMC ──────────────────────────────────────────────────────────────────────
def run_mcmc(start_params, t_data, a_data, scale_params=None):
    ndim = len(start_params)
    if scale_params is None:
        scale_params = np.array([0.2] * ndim)

    p0 = np.array([
        start_params + scale_params * np.random.randn(ndim)
        for _ in range(nwalkers)
    ])

    test_logps = np.array([log_posterior(p, t_data, a_data) for p in p0])
    n_finite   = np.sum(np.isfinite(test_logps))
    if n_finite == 0:
        print("  All initial log-posteriors are -inf. Aborting MCMC.")
        return None
    if n_finite < nwalkers:
        print(f"  Warning: only {n_finite}/{nwalkers} initial positions are finite.")

    with multiprocessing.Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, args=(t_data, a_data), pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True, store=True)

    print(f"  Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    chain   = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, flat=True)
    return samples, chain, sampler


# ── Parameter uncertainty (Laplace approximation) ─────────────────────────────
def _numerical_hessian(f, x, h=1e-4):
    """Central finite-difference Hessian of scalar function f at point x."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h; xm[i] -= h
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h**2
            else:
                xpp, xpm = x.copy(), x.copy()
                xmp, xmm = x.copy(), x.copy()
                xpp[i] += h; xpp[j] += h
                xpm[i] += h; xpm[j] -= h
                xmp[i] -= h; xmp[j] += h
                xmm[i] -= h; xmm[j] -= h
                H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h**2)
                H[j, i] = H[i, j]
    return H


def ls_uncertainties(params, t_data, a_data):
    """
    Laplace-approximation uncertainties from curvature of -log_posterior.
    Returns dict with keys k1_err, k2_err, m_err, n_err, or None on failure.
    """
    try:
        neg_logp = lambda p: -log_posterior(p, t_data, a_data)
        H = _numerical_hessian(neg_logp, params)
        cov = np.linalg.inv(H)
        if np.any(np.diag(cov) < 0):
            return None
        log_k1, log_k2, m, n, _ = params
        sigma_log_k1, sigma_log_k2, sigma_m, sigma_n = np.sqrt(np.diag(cov)[:4])
        return {
            "k1_err": 10**log_k1 * np.log(10) * sigma_log_k1,
            "k2_err": 10**log_k2 * np.log(10) * sigma_log_k2,
            "m_err":  sigma_m,
            "n_err":  sigma_n,
        }
    except Exception:
        return None


# ── LS fit plot ────────────────────────────────────────────────────────────────
def plot_ls_fit(params, t_data, a_data, sample, temp):
    os.makedirs(PLOT_DIR, exist_ok=True)
    label = f"NMR_{sample}_{temp}_km"

    log_k1, log_k2, m, n, _ = params
    k1, k2 = 10**log_k1, 10**log_k2
    a_fit = solve_model(*params[:-1], t_data, a_data)

    errs = ls_uncertainties(params, t_data, a_data)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_data, a_data, "o", ms=4, label="Data")
    if np.all(np.isfinite(a_fit)):
        ax.plot(t_data, a_fit, "-", label="KM fit")
    ax.set(title=f"{sample} {temp} — Kamal-Malkin (r={R})",
           xlabel="Time (s)", ylabel="α")
    ax.legend(loc="lower right")

    if errs:
        txt = (
            f"$k_1$ = {k1:.3e} ± {errs['k1_err']:.1e}\n"
            f"$k_2$ = {k2:.3e} ± {errs['k2_err']:.1e}\n"
            f"$m$  = {m:.3f} ± {errs['m_err']:.3f}\n"
            f"$n$  = {n:.3f} ± {errs['n_err']:.3f}"
        )
    else:
        txt = (
            f"$k_1$ = {k1:.3e}\n"
            f"$k_2$ = {k2:.3e}\n"
            f"$m$  = {m:.3f}\n"
            f"$n$  = {n:.3f}"
        )

    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{label}_ls.png", dpi=150)
    plt.close(fig)


# ── MCMC plotting ───────────────────────────────────────────────────────────────
def plot_results(samples, t_data, a_data, sample, temp):
    os.makedirs(PLOT_DIR, exist_ok=True)
    label = f"NMR_{sample}_{temp}_km"

    fig = corner.corner(samples, labels=PARAM_NAMES)
    fig.savefig(f"{PLOT_DIR}/{label}_corner.png")
    plt.close(fig)

    idx          = np.random.choice(len(samples), size=min(overlay_n, len(samples)), replace=False)
    alpha_preds  = np.array([
        solve_model(*samples[i][:-1], t_data, a_data) for i in idx
    ])
    alpha_preds  = alpha_preds[np.all(np.isfinite(alpha_preds), axis=1)]
    mean_alpha   = np.nanmean(alpha_preds, axis=0)
    lower_alpha  = np.nanpercentile(alpha_preds, 2.5,  axis=0)
    upper_alpha  = np.nanpercentile(alpha_preds, 97.5, axis=0)

    dadt_preds   = np.array([gradient(a, t_data) for a in alpha_preds])
    mean_dadt    = np.nanmean(dadt_preds, axis=0)
    lower_dadt   = np.nanpercentile(dadt_preds, 2.5,  axis=0)
    upper_dadt   = np.nanpercentile(dadt_preds, 97.5, axis=0)

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
    """Return (t_data, a_data) for one (sample, temp), or None if no data."""
    df   = pd.read_csv(DATA_CSV)
    mask = (df["sample"] == sample) & (df["temp"] == temp) & (~df["dropped"]) & df["alpha"].notna()
    grp  = df[mask].sort_values("elapsed_min")
    if grp.empty:
        return None
    t = grp["elapsed_min"].values.astype(float) * 60  # convert min → s
    a = grp["alpha"].values.astype(float)
    t = t - t[0]                          # zero-reference time
    a = np.clip(a, 1e-8, R - 1e-8)
    return t, a


def fit_ls(t_data, a_data, sample, temp):
    """Nelder-Mead LS fit. Returns (params, rss, converged)."""
    start = np.array([-5.0, -2.0, 0.5, 1.4, -4.0])
    opt   = minimize(
        lambda p: -log_posterior(p, t_data, a_data),
        start,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
    )
    params = opt.x
    a_fit  = solve_model(*params[:-1], t_data, a_data)
    rss    = float(np.sum((a_data - a_fit)**2)) if np.all(np.isfinite(a_fit)) else np.nan
    return params, rss, opt.success


def process_single(sample, temp, do_mcmc=False):
    label = f"{sample} {temp}"
    print(f"Fitting NMR {label} (KM, r={R})")

    data = load_dataset(sample, temp)
    if data is None:
        print(f"  No data for {label}. Skipping.")
        return None
    t_data, a_data = data

    # LS fit
    params, rss, converged = fit_ls(t_data, a_data, sample, temp)
    log_k1, log_k2, m, n, log_sigma = params
    k1, k2 = 10**log_k1, 10**log_k2
    print(f"  LS: k1={k1:.2e}, k2={k2:.2e}, m={m:.3f}, n={n:.3f}, RSS={rss:.4e}, converged={converged}")

    row = {
        "sample": sample, "temp": temp,
        "k1": k1, "k2": k2, "m": m, "n": n,
        "rss": rss, "converged": converged,
    }

    plot_ls_fit(params, t_data, a_data, sample, temp)

    if not do_mcmc:
        return row

    # MCMC
    scale = np.array([0.2, 0.2, 0.05, 0.05, 0.1])
    result = run_mcmc(params, t_data, a_data, scale)
    if result is None:
        return row
    samples, chain, sampler = result

    os.makedirs(NPZ_DIR, exist_ok=True)
    tag = f"{sample}_{temp}_km"
    np.savez(
        f"{NPZ_DIR}/{tag}_fitdata.npz",
        samples=samples, chain=chain,
        t_data=t_data, a_data=a_data,
        log_prob=sampler.get_log_prob(),
    )
    with open(f"{NPZ_DIR}/{tag}_sampler.pkl", "wb") as f:
        pickle.dump(sampler, f)

    # Update row with MCMC medians
    for i, name in enumerate(PARAM_NAMES):
        vals = 10**samples[:, i] if "log_" in name else samples[:, i]
        row[f"{name}_median"] = np.median(vals)
        row[f"{name}_p2_5"]   = np.percentile(vals, 2.5)
        row[f"{name}_p97_5"]  = np.percentile(vals, 97.5)

    plot_results(samples, t_data, a_data, sample, temp)
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

    if args.sample and args.temp:
        result = process_single(args.sample, args.temp, do_mcmc=args.mcmc)
        if result:
            pd.DataFrame([result]).to_csv(
                f"{OUT_DIR}/km_{args.sample}_{args.temp}.csv", index=False
            )
    else:
        # Discover all (sample, temp) combinations present in the CSV
        df    = pd.read_csv(DATA_CSV)
        tasks = df[["sample", "temp"]].drop_duplicates().values.tolist()

        all_results = []
        for sample, temp in tasks:
            row = process_single(sample, temp, do_mcmc=args.mcmc)
            if row:
                all_results.append(row)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{OUT_DIR}/km_results.csv", index=False)
        print(f"\nResults saved to {OUT_DIR}/km_results.csv")
        print(results_df[["sample", "temp", "k1", "k2", "m", "n", "rss", "converged"]].to_string(index=False))
