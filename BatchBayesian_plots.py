import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import corner
import argparse
from scipy.integrate import solve_ivp
from numpy import gradient
from datetime import datetime
import time
import traceback

from mcmc_config import burnin, stride, overlay_n, nwalkers, nsteps

def solve_model(log_k1, log_k2, m, n, r, t_data, eps=1e-10):
    """
    Plotting-specific model solver. Uses a fixed small a0 and does not depend on a_data.
    """
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    a0 = [eps]

    def ode_rhs(t, a):
        a = np.clip(a, eps, r - eps)
        return (k1 + k2 * a**m) * (1 - a)**(n/2) * (r - a)**(n/2)

    try:
        sol = solve_ivp(ode_rhs, [t_data[0], t_data[-1]], a0, t_eval=t_data,
                        method='LSODA', rtol=1e-6, atol=1e-8)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full_like(t_data, np.nan)
        return sol.y[0]
    except Exception as e:
        print(f"[{label}] ODE solve failed: {e}")
        return np.full_like(t_data, np.nan)

def plot_posterior_overlay(alpha_preds, t_data, a_data, label, outdir):
    # Y-axis limits based on data
    y_min = np.min(a_data)
    y_max = np.max(a_data)
    margin = 0.05 * (y_max - y_min)
    y_lim = (y_min - margin, y_max + margin)

    # Median fit
    median_fit = np.nanmedian(alpha_preds, axis=0)

    # Plot all posterior fits
    plt.figure()
    for a_fit in alpha_preds:
        plt.plot(t_data, a_fit, color='red', alpha=0.1)

    # Overlay data and median
    plt.plot(t_data, median_fit, 'r-', lw=2, label="Median Fit")
    plt.plot(t_data, a_data, 'k.', label="Data")

    plt.xlabel("Time")
    plt.ylabel("α(t)")
    plt.title(f"{label} Posterior Overlay")
    plt.ylim(y_lim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_overlay.png", dpi=300)
    plt.close()

def plot_alpha_confidence_band(alpha_preds, t_data, a_data, label, outdir):
    # Summary stats from posterior predictions
    median_alpha = np.nanmedian(alpha_preds, axis=0)
    lower_alpha = np.nanpercentile(alpha_preds, 2.5, axis=0)
    upper_alpha = np.nanpercentile(alpha_preds, 97.5, axis=0)

    # Determine y-axis limits based on a_data only
    y_min = np.min(a_data)
    y_max = np.max(a_data)
    margin = 0.05 * (y_max - y_min)
    y_lim = (y_min - margin, y_max + margin)

    # Plot
    plt.figure()
    plt.plot(t_data, a_data, 'k.', label='Data')
    plt.plot(t_data, median_alpha, 'r-', label='Median Fit')
    plt.fill_between(t_data, lower_alpha, upper_alpha, color='red', alpha=0.3, label='95% CI')

    plt.xlabel("Time")
    plt.ylabel("α(t)")
    plt.title(f"{label} α(t) with 95% CI")
    plt.ylim(y_lim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_alpha_ci.png", dpi=300)
    plt.close()

def plot_dadt_vs_alpha(alpha_preds, t_data, a_data, label, outdir):
    # Posterior derivatives
    dadt_preds = np.array([gradient(alpha, t_data) for alpha in alpha_preds])
    median_dadt = np.nanmedian(dadt_preds, axis=0)
    lower_dadt = np.nanpercentile(dadt_preds, 2.5, axis=0)
    upper_dadt = np.nanpercentile(dadt_preds, 97.5, axis=0)
    median_alpha = np.nanmedian(alpha_preds, axis=0)

    # Raw data derivative
    dadt_data = gradient(a_data, t_data)

    # Use only data to determine y-axis limits
    y_min = np.min(dadt_data)
    y_max = np.max(dadt_data)
    margin = 0.1 * (y_max - y_min)
    y_lim = (y_min - margin, y_max + margin)

    # Plot
    plt.figure()
    plt.plot(median_alpha, median_dadt, 'r-', label="Median Posterior")
    plt.fill_between(median_alpha, lower_dadt, upper_dadt, color='red', alpha=0.3, label="95% CI")
    plt.plot(a_data, dadt_data, 'k.', label="Raw Data")

    plt.xlabel("α")
    plt.ylabel("dα/dt")
    plt.title(f"{label} dα/dt vs α")
    plt.ylim(y_lim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/{label}_dadt_alpha.png", dpi=300)
    plt.close()

def plot_corner(samples, label, outdir):
    fig = corner.corner(samples, labels=["log_k1", "log_k2", "m", "n", "r", "log_sigma"])
    fig.savefig(f"{outdir}/{label}_corner.png", dpi=300)
    plt.close(fig)

def plot_chains(chain, label, outdir):
    nsteps, nwalkers, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    param_names = ["log_k1", "log_k2", "m", "n", "r", "log_sigma"]
    for i in range(ndim):
        ax = axes[i]
        for walker in range(nwalkers):
            ax.plot(chain[:, walker, i], alpha=0.3)
        ax.set_ylabel(param_names[i])
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} MCMC Chains")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_chains.png", dpi=300)
    plt.close(fig)

def make_summary_grid(label, outdir="fit_plots"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"{label} Summary", fontsize=16)

    def load_and_show(ax, filepath, title=None):
        if os.path.exists(filepath):
            img = mpimg.imread(filepath)
            ax.imshow(img)
            ax.axis('off')
            if title:
                ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=12)
            ax.axis('off')

    prefix = f"{outdir}/{label}"
    load_and_show(axes[0, 0], f"{prefix}_overlay.png", "Posterior Overlay")
    load_and_show(axes[0, 1], f"{prefix}_alpha_ci.png", "α(t) with 95% CI")
    load_and_show(axes[1, 0], f"{prefix}_dadt_alpha.png", "dα/dt vs α")
    load_and_show(axes[1, 1], f"{prefix}_chains.png", "MCMC Chains")
    load_and_show(axes[2, 0], f"{prefix}_corner.png", "Corner Plot")

    # Leave bottom right blank but add timestamp
    axes[2, 1].axis('off')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    axes[2, 1].text(1.0, -0.1, f"Generated {timestamp}", ha='right', va='top', transform=axes[2, 1].transAxes, fontsize=10, color='gray')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{prefix}_summary.png", dpi=300)
    plt.close(fig)

def make_plots(samples, chain, t_data, a_data, label, outdir="fit_plots", overlay_n=overlay_n, burnin=burnin, stride=stride):
    import time  # in case it's not already imported
    os.makedirs(outdir, exist_ok=True)
    start_time = time.time()
    print(f"📊 Starting plots for {label}", flush=True)

    # Apply burn-in and thinning
    samples = samples[burnin::stride]
    chain = chain[burnin::stride]  # (steps, walkers, ndim)
    chain = np.transpose(chain, (1, 0, 2))  # → (walkers, steps, ndim) for plotting

    # Randomly sample posterior draws
    idx = np.random.choice(len(samples), size=min(overlay_n, len(samples)), replace=False)
    subset = samples[idx]

    # Solve and filter
    alpha_preds = []
    filtered_subset = []

    for p in subset:
        a_fit = solve_model(*p[:5], t_data)

        max_ok = np.max(a_fit) < 1.05 * np.max(a_data)
        min_ok = np.min(a_fit) > 0
        final_close = np.abs(a_fit[-1] - a_data[-1]) < 0.05 * np.max(a_data)
        monotonic = np.all(np.diff(a_fit) >= -0.01)

        if (
            np.all(np.isfinite(a_fit)) and
            max_ok and
            min_ok and
            final_close and
            monotonic
        ):
            alpha_preds.append(a_fit)
            filtered_subset.append(p)

    alpha_preds = np.array(alpha_preds)
    filtered_subset = np.array(filtered_subset)

    print(f"[{label}] Posterior fit filtering: {len(filtered_subset)}/{len(subset)} curves retained "
          f"({len(subset) - len(filtered_subset)} rejected).", flush=True)

    # Plot all outputs using filtered data
    t0 = time.time(); plot_posterior_overlay(alpha_preds, t_data, a_data, label, outdir); print(f"⏱ overlay: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time(); plot_alpha_confidence_band(alpha_preds, t_data, a_data, label, outdir); print(f"⏱ alpha CI: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time(); plot_dadt_vs_alpha(alpha_preds, t_data, a_data, label, outdir); print(f"⏱ dα/dt vs α: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time(); plot_chains(chain, label, outdir); print(f"⏱ chains: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time(); plot_corner(samples, label, outdir); print(f"⏱ corner: {time.time() - t0:.2f}s", flush=True)
    t0 = time.time(); make_summary_grid(label, outdir); print(f"⏱ summary grid: {time.time() - t0:.2f}s", flush=True)

    print(f"✅ Finished all plots for {label} in {time.time() - start_time:.2f}s", flush=True)

def load_npz_files(directory):
    for file in os.listdir(directory):
        if file.endswith("_fitdata.npz"):
            yield os.path.join(directory, file)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Plot MCMC results from saved .npz files.")
        parser.add_argument("file", nargs="?", default=None,
                            help="Optional path to a single .npz file. If omitted, all *_fitdata.npz files in mcmc_samples/ will be processed.")
        parser.add_argument("--burnin", type=int, default=None, help="Burn-in steps to discard")
        parser.add_argument("--stride", type=int, default=None, help="Thinning stride")
        args = parser.parse_args()

        input_dir = "mcmc_samples"

        if args.file:
            # Just one file
            file_path = args.file
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}", flush=True)
            else:
                label = os.path.basename(file_path).replace("_fitdata.npz", "")
                data = np.load(file_path)
                samples = data["samples"]
                chain = data["chain"]
                t_data = data["t_data"]
                a_data = data["a_data"]
                make_plots(samples, chain, t_data, a_data, label,
                           burnin=args.burnin, stride=args.stride)
        else:
            # Run on all files in mcmc_samples/
            for file_path in load_npz_files(input_dir):
                label = os.path.basename(file_path).replace("_fitdata.npz", "")
                data = np.load(file_path)
                samples = data["samples"]
                chain = data["chain"]
                t_data = data["t_data"]
                a_data = data["a_data"]
                make_plots(samples, chain, t_data, a_data, label,
                           burnin=args.burnin, stride=args.stride)

    except Exception as e:
        print("💥 An error occurred during plotting:", flush=True)
        traceback.print_exc()

