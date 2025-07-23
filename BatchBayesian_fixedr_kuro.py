# Core Python
import os
import sys
import argparse
import pickle
import multiprocessing
from multiprocessing import freeze_support
from datetime import datetime

# Scientific computing
import numpy as np
import pandas as pd
from numpy import gradient
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# MCMC & plotting
import emcee
import corner
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Progress tracking
from tqdm import tqdm

# Config
from mcmc_config import burnin, stride, overlay_n, nwalkers, nsteps


LOG_FILE = "posterior_debug.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log_debug(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\r\n")

# param_names = ['log_k1', 'log_k2', 'm', 'n', 'r', 'log_sigma']
param_names = ['log_k1', 'log_k2', 'm', 'n', 'log_sigma']
samples_list = ['EDA', 'DAP', 'DAB']
dsc_temps = [25, 33, 50, 60, 80, 100]
nmr_temps = [25, 33, 40]

# Data loading
mat = loadmat('epoxy_data.mat')

try:
    fit_csv = pd.read_csv("fit_results/combined_results.csv", engine='python')
    fit_csv = fit_csv.copy(deep=True)  # ensure safe read-only behavior
except Exception as e:
    print(f"[Warning] Could not load combined_results.csv for initial guesses: {e}")
    fit_csv = None

def ode_rhs(t, a, log_k1, log_k2, m, n, r, eps=1e-10):
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    a = np.clip(a, eps, r - eps)
    return (k1 + k2 * a**m) * (1 - a)**(n/2) * (r - a)**(n/2)

def solve_model(log_k1, log_k2, m, n, r, t_data, a_data, eps=1e-10):
    try:
        a0 = np.clip(a_data[0], eps, r - eps)
        sol = solve_ivp(ode_rhs, [t_data[0], t_data[-1]], [a0], t_eval=t_data,
                        args=(log_k1, log_k2, m, n, r), method='LSODA',
                        rtol=1e-6, atol=1e-8)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            log_debug(f"[ODE FAIL] success={sol.success}, finite={np.all(np.isfinite(sol.y))}, log_k1={log_k1}, log_k2={log_k2}, m={m}, n={n}, r={r}, a0={a0}, t_span=({t_data[0]}, {t_data[-1]})\r\n")
            return np.full_like(t_data, np.nan)
        return np.clip(sol.y[0], 1e-6, r - 1e-6)
    except Exception as e:
        log_debug(f"[SOLVE EXCEPTION] log_k1={log_k1}, log_k2={log_k2}, m={m}, n={n}, r={r} => {e}\r\n")
        return np.full_like(t_data, np.nan)
    

def log_prior(params, t_data, a_data, r):
    log_k1, log_k2, m, n, log_sigma = params
    # Loosened prior bounds to allow exploration
    if not (-10 < log_k1 < -0.5 and
            -10 < log_k2 < -0.5 and
             0 < m < 3 and
             0 < n < 3 and
            -12 < log_sigma < 0):
        log_debug(f"[PRIOR REJECT] Params: {params}\r\n")
        return -np.inf
    return 0.0

def log_likelihood(params, t_data, a_data, r):
    log_k1, log_k2, m, n, log_sigma = params
    # print(f"Inside log_likelihood: {log_k1=}, {log_k2=}, {m=}, {n=}, {log_sigma=}, {r=}")
    if r <= 0 or not np.isfinite(r):
        return -np.inf  # 🚫 invalid r

    try:
        a_fit = solve_model(log_k1, log_k2, m, n, r, t_data, a_data)
        a_fit = np.clip(a_fit, 1e-6, r - 1e-6)
    except Exception as e:
        log_debug(f"[SOLVE EXCEPTION] {params} => {e}\r\n")
        return -np.inf

    if not np.all(np.isfinite(a_fit)):
        log_debug(f"[LIKE REJECT] NaNs in a_fit — Params: {params}\r\n")
        return -np.inf
    if np.any(a_fit <= 0):
        log_debug(f"[LIKE REJECT] a_fit <= 0 — min={np.min(a_fit)}, Params: {params}\r\n")
        return -np.inf
    if np.any(a_fit >= r):
        log_debug(f"[LIKE REJECT] a_fit >= r — max={np.max(a_fit)}, r={r}, Params: {params}\r\n")
        return -np.inf

    sigma = 10 ** log_sigma
    residual = (a_data - a_fit) / sigma
    return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2))


def log_posterior(params, t_data, a_data, r):
    lp = log_prior(params, t_data, a_data, r)
    if not np.isfinite(lp):
        log_debug(f"[POST REJECT] Prior -inf — Params: {params}\r\n")
        return -np.inf
    ll = log_likelihood(params, t_data, a_data, r)
    if not np.isfinite(ll):
        log_debug(f"[POST REJECT] Likelihood -inf — Params: {params}\r\n")
        return -np.inf
    log_debug(f"[POST OK] Params: {params} => logP = {lp + ll:.3f}\r\n")
    return lp + ll


def plot_log_posterior_slices(best_params, t_data, a_data, r, param_names):
    print("\r\n🩺 Plotting posterior slices...")
    os.makedirs("diagnostic_plots", exist_ok=True)

    for i, name in enumerate(param_names):
        if name in ['m', 'n']:  # optional: skip for now
            continue

        print(f"\r\n🔬 Scanning posterior slice for {name}")
        label = name
        if "log_" in name:
            param_range = np.linspace(best_params[i] - 1, best_params[i] + 1, 200)
        else:
            param_range = np.linspace(
                max(0.01, best_params[i] * 0.5),
                best_params[i] * 1.5,
                200
            )

        log_post_vals = []
        for val in param_range:
            trial = best_params.copy()
            trial[i] = val
            lp = log_posterior(trial, t_data, a_data, r)

            # 🔍 Diagnostic print before appending
            print(f"  {label} = {val:.4f}, log_posterior = {lp:.3f}" if np.isfinite(lp) else f"  {label} = {val:.4f}, log_posterior = -inf")

            log_post_vals.append(lp if np.isfinite(lp) else np.nan)

        # Plot
        plt.figure()
        plt.plot(param_range, log_post_vals)
        plt.title(f"Posterior Slice: {name}")
        plt.xlabel(name)
        plt.ylabel("log posterior")
        plt.axvline(best_params[i], color='r', linestyle='--', label='Start')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"diagnostic_plots/posterior_slice_{name}.png")
        plt.close()

        # 🔍 Per-parameter log-posterior range
        finite_vals = [v for v in log_post_vals if np.isfinite(v)]
        if finite_vals:
            print(f"  → log_posterior range: {min(finite_vals):.1f} to {max(finite_vals):.1f}")
        else:
            print("  ⚠️ All log_posterior values are -inf or NaN.")

def scan_log_posterior_grid(log_k1_vals, log_k2_vals, fixed_params, t_data, a_data, r):
    m, n, log_sigma = fixed_params
    Z = np.full((len(log_k1_vals), len(log_k2_vals)), np.nan)

    print("🧭 Running posterior grid scan...")
    for i, log_k1 in enumerate(tqdm(log_k1_vals, desc="Scanning log_k1")):
        if i % 10 == 0:
            print(f"🔍 Row {i}/{len(log_k1_vals)}: log_k1 = {log_k1:.3f}")
        for j, log_k2 in enumerate(log_k2_vals):
            params = [log_k1, log_k2, m, n, log_sigma]
            try:
                val = log_posterior(params, t_data, a_data, r)
                Z[i, j] = val if np.isfinite(val) else np.nan
            except Exception as e:
                Z[i, j] = np.nan

    os.makedirs("diagnostic_plots", exist_ok=True)

    # Save .npz
    np.savez("diagnostic_plots/log_posterior_grid.npz", log_k1=log_k1_vals, log_k2=log_k2_vals, log_post=Z)

    # Find max
    max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    max_log_k1 = log_k1_vals[max_idx[0]]
    max_log_k2 = log_k2_vals[max_idx[1]]
    max_val = Z[max_idx]
    print(f"🎯 Max log posterior: log_k1={max_log_k1:.3f}, log_k2={max_log_k2:.3f}, val={max_val:.2f}")

    # 2D Heatmap with overlay
    plt.figure(figsize=(8, 6))
    plt.contourf(log_k1_vals, log_k2_vals, Z.T, levels=100, cmap='viridis')
    plt.colorbar(label="log posterior")
    plt.scatter([max_log_k1], [max_log_k2], color='red', label='Max log posterior')
    plt.xlabel("log_k1")
    plt.ylabel("log_k2")
    plt.title("Posterior Heatmap with Max Overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig("diagnostic_plots/posterior_heatmap_with_max.png")
    plt.close()

    # 3D Surface Plot with slices
    K1, K2 = np.meshgrid(log_k1_vals, log_k2_vals, indexing='ij')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(221, projection='3d')
    surf = ax.plot_surface(K1, K2, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Posterior Surface (Zoomed)')
    ax.set_xlabel('log_k1')
    ax.set_ylabel('log_k2')
    ax.set_zlabel('log posterior')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.scatter(max_log_k1, max_log_k2, max_val, color='red', s=50)
    ax.text(max_log_k1, max_log_k2, max_val, f'\nlog_k1={max_log_k1:.2f}\nlog_k2={max_log_k2:.2f}', color='red')

    # Slice 1: log posterior vs log_k1 (log_k2 fixed at max)
    ax2 = fig.add_subplot(222)
    ax2.plot(log_k1_vals, Z[:, max_idx[1]], label=f'log_k2 = {max_log_k2:.2f}')
    ax2.axvline(max_log_k1, color='red', linestyle='--')
    ax2.set_title("Posterior Slice (log_k1)")
    ax2.set_xlabel("log_k1")
    ax2.set_ylabel("log posterior")

    # Slice 2: log posterior vs log_k2 (log_k1 fixed at max)
    ax3 = fig.add_subplot(223)
    ax3.plot(log_k2_vals, Z[max_idx[0], :], label=f'log_k1 = {max_log_k1:.2f}')
    ax3.axvline(max_log_k2, color='red', linestyle='--')
    ax3.set_title("Posterior Slice (log_k2)")
    ax3.set_xlabel("log_k2")
    ax3.set_ylabel("log posterior")

    plt.tight_layout()
    plt.savefig("diagnostic_plots/zoomed_surface_with_slices.png")
    plt.close()

    # Export best start params for walker init (optional helper file)
    best_params = np.array([max_log_k1, max_log_k2, m, n, log_sigma])
    np.save("diagnostic_plots/grid_start_params.npy", best_params)

    print("✅ Posterior grid scan complete. Results and best guess saved to 'diagnostic_plots/'")


def run_mcmc(start_params, t_data, a_data, r, scale_params=None):
    ndim = len(start_params)
    if scale_params is None:
        scale_params = [1e-4] * ndim

    p0 = np.array([
        start_params + scale_params * np.random.randn(ndim)
        for _ in range(nwalkers)
    ])

    # 🔍 Sanity check: test initial log-posteriors
    test_logps = np.array([log_posterior(p, t_data, a_data, r) for p in p0])
    n_finite = np.sum(np.isfinite(test_logps))

    if n_finite == 0:
        print("❌ All initial log-posteriors are -inf or NaN. Aborting MCMC.")
        return None
    elif n_finite < len(p0):
        print(f"⚠️ Only {n_finite}/{len(p0)} initial positions are finite.")

    with multiprocessing.Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior, args=(t_data, a_data, r), pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True, store=True)

    print("✅ Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    chain = sampler.get_chain()
    samples = sampler.get_chain(discard=burnin, flat=True)
    return samples, chain, sampler

def plot_results(samples, t_data, a_data, method, sample, temp):
    os.makedirs("fit_plots", exist_ok=True)
    label = f"{method}_{sample}_{temp}C"

    # Corner plot
    # fig = corner.corner(samples, labels=["log_k1", "log_k2", "m", "n", "r", "log_sigma"])
    fig = corner.corner(samples, labels=["log_k1", "log_k2", "m", "n", "log_sigma"])
    fig.savefig(f"fit_plots/{label}_corner.png")
    plt.close(fig)

    # Prediction samples
    idx = np.random.choice(len(samples), size=min(200, len(samples)), replace=False)
    alpha_preds = np.array([
        solve_model(*samples[i][:-1], t_data, a_data)
        for i in idx
    ])
    alpha_preds = alpha_preds[np.all(np.isfinite(alpha_preds), axis=1)]

    mean_alpha = np.nanmean(alpha_preds, axis=0)
    lower_alpha = np.nanpercentile(alpha_preds, 2.5, axis=0)
    upper_alpha = np.nanpercentile(alpha_preds, 97.5, axis=0)

    # Derivatives
    dadt_preds = np.array([gradient(alpha, t_data) for alpha in alpha_preds])
    mean_dadt = np.nanmean(dadt_preds, axis=0)
    lower_dadt = np.nanpercentile(dadt_preds, 2.5, axis=0)
    upper_dadt = np.nanpercentile(dadt_preds, 97.5, axis=0)

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # α(t) vs t
    axes[0].plot(t_data, a_data, 'o', label="Data")
    axes[0].plot(t_data, mean_alpha, label="Median Fit")
    axes[0].fill_between(t_data, lower_alpha, upper_alpha, alpha=0.3, label="95% CI")
    axes[0].set_title(f"{label}: α(t)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("α")
    axes[0].legend()

    # dα/dt vs α
    axes[1].plot(mean_alpha, mean_dadt, label="Median")
    axes[1].fill_between(mean_alpha, lower_dadt, upper_dadt, alpha=0.3, label="95% CI")
    axes[1].set_title(f"{label}: dα/dt vs α")
    axes[1].set_xlabel("α")
    axes[1].set_ylabel("dα/dt")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"fit_plots/{label}_combined.png")
    plt.close(fig)

def process_single(task):
    method, sample, temp = task
    print(f"Fitting {method} {sample} at {temp} °C")

    dataset_name = 'NMR' if method == 'NMR' else sample
    index_map = {'NMR': [25, 33, 40], 'DSC': [25, 33, 50, 60, 80, 100]}
    offset = {'EDA': 0, 'DAP': 3, 'DAB': 6}.get(sample, 0)
    ii = offset + index_map[method].index(temp) if method == 'NMR' else index_map['DSC'].index(temp)

    try:
        t_data = np.squeeze(mat[dataset_name][0, 0]['clean_time'][0, ii])
        a_data = np.squeeze(mat[dataset_name][0, 0]['clean_alpha_unscaled'][0, ii])
        t_data -= t_data[0]
        a_data = np.clip(a_data, 1e-6, 1 - 1e-6)
        imax = np.argmax(a_data)
        t_data, a_data = t_data[:imax + 1], a_data[:imax + 1]
    except Exception:
        print(f"Dataset {method}_{sample}_{temp}C not found.")
        return None

    r = np.max(a_data)  # always fix r from data
    grid_param_file = "diagnostic_plots/grid_start_params.npy"

    # Get initial params
    if os.path.exists(grid_param_file):
        print("📦 Using grid scan start params")
        start = np.load(grid_param_file)
        scale_params = np.array([0.1, 0.1, 0.02, 0.02, 0.05])
    elif fit_csv is not None:
        row = fit_csv[(fit_csv["Method"] == method) & (fit_csv["Sample"] == sample) & (fit_csv["Temperature"] == temp)]
        if not row.empty:
            row = row.iloc[0]
            # Estimate safe log_sigma from clipped std
            std_clip = np.std(a_data - np.clip(a_data, 1e-6, 1 - 1e-6))
            log_sigma_init = np.log10(std_clip) if std_clip > 1e-12 else -4

            start = [
                np.log10(row["Fit_k1"]),
                np.log10(row["Fit_k2"]),
                row["Fit_m"],
                row["Fit_n"],
                log_sigma_init
            ]
            try:
                scale_params = np.array([
                    row["Unc_k1"] / row["Fit_k1"] / np.log(10) if row["Fit_k1"] > 0 else 0.2,
                    row["Unc_k2"] / row["Fit_k2"] / np.log(10) if row["Fit_k2"] > 0 else 0.2,
                    row["Unc_m"], row["Unc_n"], 0.2
                ])
            except Exception as e:
                print(f"[Warning] Missing uncertainty values: {e}")
                scale_params = np.array([0.2] * 5)
            scale_params = np.maximum(scale_params, 0.05)
        else:
            print(f"[Fallback] No CSV match for {method}_{sample}_{temp}C.")
            start = [-5, -2, 0.5, 1.4, -4]
            scale_params = np.array([0.2] * 5)
    else:
        start = [-5, -2, 0.5, 1.4, -4]
        scale_params = np.array([0.2] * 5)

    print(f"🔧 MCMC init: start = {start}")
    print(f"🔧 MCMC init: scale_params = {scale_params}")

    opt = minimize(lambda p: -log_posterior(p, t_data, a_data, r), start, method='Nelder-Mead', options={'maxiter': 1000})
    init_params = opt.x if opt.success and not np.any(np.isnan(opt.x)) else start
    print("r (fixed):", r)

    plot_log_posterior_slices(np.array(start), t_data, a_data, r, param_names)
    result = run_mcmc(init_params, t_data, a_data, r, scale_params)
    if result is None:
        print(f"🚫 Skipping {method}_{sample}_{temp}C due to failed MCMC init.")
        return None
    samples, chain, sampler = result
    
    label = f"{method}_{sample}_{temp}C"
    os.makedirs("mcmc_samples", exist_ok=True)

    np.savez(f"mcmc_samples/{label}_fitdata.npz", samples=samples, chain=chain, t_data=t_data, a_data=a_data, log_prob=sampler.get_log_prob())
    with open(f"mcmc_samples/{label}_sampler.pkl", "wb") as f:
        pickle.dump(sampler, f)

    summary = {}
    for i, name in enumerate(param_names):
        vals = 10**samples[:, i] if "log_" in name else samples[:, i]
        summary[f"{name}_median"] = np.median(vals)
        summary[f"{name}_lower"] = np.percentile(vals, 2.5)
        summary[f"{name}_upper"] = np.percentile(vals, 97.5)

    return {"Method": method, "Sample": sample, "Temp_C": temp, **summary}

if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Run MCMC fit for one or all datasets.")
    parser.add_argument("method", nargs="?", choices=["NMR", "DSC"], help="Measurement method")
    parser.add_argument("sample", nargs="?", choices=["EDA", "DAP", "DAB"], help="Sample type")
    parser.add_argument("temp", nargs="?", type=int, help="Temperature in Celsius")
    parser.add_argument("--grid_scan", action="store_true", help="Run posterior grid scan instead of MCMC")
    args = parser.parse_args()

    output_dir = "fit_results"
    os.makedirs(output_dir, exist_ok=True)

    if args.grid_scan:
        # Prepare the data
        method = args.method
        sample = args.sample
        temp = args.temp

        print(f"🗺️  Running posterior grid scan for {method} {sample} at {temp} °C")

        # Load data just like in process_single
        dataset_name = sample if method != "NMR" else "NMR"
        if method == "NMR":
            offset = {"EDA": 0, "DAP": 3, "DAB": 6}[sample]
            ii = offset + [25, 33, 40].index(temp)
        else:
            ii = [25, 33, 50, 60, 80, 100].index(temp)

        t_data = np.squeeze(mat[dataset_name][0, 0]['clean_time'][0, ii])
        a_data = np.squeeze(mat[dataset_name][0, 0]['clean_alpha_unscaled'][0, ii])
        t_data = t_data - t_data[0]
        a_data = np.clip(a_data, 1e-6, 1 - 1e-6)
        r = np.max(a_data)

        # Define grid and fixed parameters
        log_k1_vals = np.linspace(-6, -3, 100)
        log_k2_vals = np.linspace(-4, -2, 100)
        fixed_params = (0.5, 1.4, -4.0)  # Replace with your default m, n, log_sigma

        scan_log_posterior_grid(log_k1_vals, log_k2_vals, fixed_params, t_data, a_data, r)
        sys.exit(0)  # Exit after grid scan

    if args.method and args.sample and args.temp:
        # Run one specific dataset
        result = process_single((args.method, args.sample, args.temp))
        if result:
            results_df = pd.DataFrame([result])
            results_df.to_csv(f"{output_dir}/fit_{args.method}_{args.sample}_{args.temp}C.csv", index=False)
    else:
        # Run full batch
        tasks = [(method, sample, temp)
                 for sample in samples_list
                 for method, temps in [('DSC', dsc_temps), ('NMR', nmr_temps)]
                 for temp in temps]

        all_results = []
        for task in tasks:
            result = process_single(task)
            if result:
                all_results.append(result)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv('parameter_estimates.csv', index=False)
        print("All results saved to parameter_estimates.csv")