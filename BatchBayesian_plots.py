import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import corner
import argparse
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from mcmc_config import overlay_n as default_overlay_n, burnin as default_burnin, stride as default_stride
from numpy import gradient
from datetime import datetime
import time
import traceback
import csv

from mcmc_config import overlay_n, nwalkers, nsteps

def solve_model(log_k1, log_k2, m, n, r, t_data, a_data, eps=1e-10):
    """
    Plotting-specific model solver. Uses a fixed small a0 and does not depend on a_data.
    """
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    assert r > 0, f"Invalid r: {r}"
    a0 = [np.clip(a_data[0], eps, r - eps)]  # ✅ start from observed α(0)

    def ode_rhs(t, a):
        a = np.clip(a, eps, r - eps)
        rate = (k1 + k2 * a**m) * (1 - a)**(n/2) * (r - a)**(n/2)
        if not np.isfinite(rate):
            print(f"💥 NaN in rate: a={a}, params=({log_k1}, {log_k2}, {m}, {n}, {r})")
            return 0.0  # or np.nan if you want it to break
        return rate

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
    # Clip out extreme outliers
    lower_bounds = np.quantile(samples, 0.01, axis=0)
    upper_bounds = np.quantile(samples, 0.99, axis=0)
    mask = np.all((samples >= lower_bounds) & (samples <= upper_bounds), axis=1)
    filtered = samples[mask]

    # Plot with nice range control
    ranges = [
        (np.percentile(filtered[:, i], 0.5), np.percentile(filtered[:, i], 99.5))
        for i in range(filtered.shape[1])
    ]

    fig = corner.corner(filtered, labels=["log_k1", "log_k2", "m", "n", "log_sigma"], range=ranges)
    fig.savefig(f"{outdir}/{label}_corner.png", dpi=300)
    plt.close(fig)


def plot_chains(chain, label, outdir):
    nwalkers, nsteps, ndim = chain.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    param_names = ["log_k1", "log_k2", "m", "n", "log_sigma"]

    for i in range(ndim):
        ax = axes[i]
        for w in range(nwalkers):
            ax.plot(chain[w, :, i], alpha=0.3)
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

def make_plots(samples, chain, t_data, a_data, r, label, outdir="fit_plots", overlay_n=None, burnin=None, stride=None):
    if burnin is None:
        burnin = default_burnin
    if stride is None:
        stride = default_stride
    if overlay_n is None:
        overlay_n = default_overlay_n
    os.makedirs(outdir, exist_ok=True)
    start_time = time.time()
    print(f"📊 Starting plots for {label}", flush=True)

    # Apply burn-in and thinning
    print(f"📦 samples shape before: {samples.shape}")
    samples = samples[burnin::stride]
    print(f"📦 samples shape after burnin/stride: {samples.shape}")
    chain = chain[burnin::stride]  # (steps, walkers, ndim)
    chain = np.transpose(chain, (1, 0, 2))  # → (walkers, steps, ndim) for plotting

    walker_means = np.mean(chain, axis=1)  # shape: (nwalkers, ndim)

    # Compute ensemble mean and cov
    ensemble_mean = np.mean(walker_means, axis=0)
    ensemble_cov = np.cov(walker_means.T)
    inv_cov = np.linalg.pinv(ensemble_cov)  # robust inverse

    # Compute Mahalanobis distance per walker
    dists = np.array([
        mahalanobis(walker_means[i], ensemble_mean, inv_cov)
        for i in range(len(walker_means))
    ])

    # Filter walkers: keep those within N std of the center
    cutoff = 3.0
    walker_ok = dists < cutoff
    n_total = len(walker_ok)
    n_kept = np.sum(walker_ok)

    for i, ok in enumerate(walker_ok):
        if not ok:
            print(f"🚫 Walker {i} removed: Mahalanobis dist = {dists[i]:.2f} > {cutoff}")

    print(f"📊 Kept {n_kept} / {n_total} walkers")

    # Filter chain and rebuild samples
    chain = chain[walker_ok]  # (n_kept, nsteps, ndim)
    samples = chain.transpose(1, 0, 2).reshape(-1, chain.shape[-1])
    # filtered_samples = samples  # ✅ Use this for posterior summary and corner plot
    
    param_names = ["log_k1", "log_k2", "m", "n", "log_sigma"]
    print(f"\n📊 Per-walker parameter stats (after burnin/stride/removing stuck walkers):")
    for w in range(chain.shape[0]):
        print(f"\n🧍 Walker {w}:")
        for i, name in enumerate(param_names):
            mean = np.mean(chain[w, :, i])
            std = np.std(chain[w, :, i])
            print(f"   {name:10s} → mean = {mean:+.5f}, std = {std:.2e}")

    # Randomly sample posterior draws
    idx = np.random.choice(len(samples), size=min(overlay_n, len(samples)), replace=False)
    subset = samples[idx]

    print(f"🧪 overlay_n = {overlay_n}, subset.shape = {subset.shape}")

    # Solve and filter
    alpha_preds = []
    filtered_subset = []

    for p in subset:
        log_k1, log_k2, m, n, log_sigma = p
        a_fit = solve_model(log_k1, log_k2, m, n, r, t_data, a_data)
        # print(f"▶ solve_model returned: {a_fit}")
        # print(f"▶ isfinite: {np.all(np.isfinite(a_fit))}, any nans? {np.any(np.isnan(a_fit))}")

        if not np.all(np.isfinite(a_fit)):
            print(f"💩 NaNs in a_fit | params: {p}")
            continue

        # print(f"✅ a_fit: min={np.min(a_fit):.4f}, max={np.max(a_fit):.4f}, final={a_fit[-1]:.4f}")
        # print(f"✅ a_data: final={a_data[-1]:.4f}, max={np.max(a_data):.4f}")

        max_ok = np.max(a_fit) < 1.05 * np.max(a_data)
        min_ok = np.min(a_fit) > 1e-8
        final_close = np.abs(a_fit[-1] - a_data[-1]) < 0.05 * np.max(a_data)
        monotonic = np.all(np.diff(a_fit) >= -0.01)

        if not max_ok:
            print("❌ Failed max_ok")
        if not min_ok:
            print("❌ Failed min_ok")
        if not final_close:
            print("❌ Failed final_close")
        if not monotonic:
            print("❌ Failed monotonic")

        if max_ok and min_ok and final_close and monotonic:
            alpha_preds.append(a_fit)
            filtered_subset.append(p)

        # === stricter filtering ===
        # if (
        #     np.all(np.isfinite(a_fit)) and
        #     max_ok and
        #     min_ok and
        #     final_close and
        #     monotonic
        # ):
        #     alpha_preds.append(a_fit)
        #     filtered_subset.append(p)

        # === looser filtering ===
        # if np.all(np.isfinite(a_fit)):
        #     alpha_preds.append(a_fit)
        #     filtered_subset.append(p)

    print(f"✅ Kept {len(alpha_preds)} / {len(subset)} curves")

    if len(alpha_preds) == 0:
        print(f"[{label}] ❌ No valid posterior curves for {label}. Plotting raw α(t) only.")
        plt.figure()
        plt.plot(t_data, a_data, 'ko', label='Observed α(t)')
        plt.title(f"{label} — Raw Data Only")
        plt.xlabel("Time")
        plt.ylabel("α(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/{label}_raw_data_only.png", dpi=300)
        return
        # ---------------------------------------
        # 🔥 DEAR FUTURE ME:
        # This return used to be outside the `if`
        # And it ABSOLUTELY FUCKING RUINED EVERYTHING
        # Don't trust your goddamn indentation.
        # ---------------------------------------

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

    # === CI width diagnostic ===
    param_names = ["log_k1", "log_k2", "m", "n", "log_sigma"]
    samples = np.array(filtered_subset)

    print(f"\n📊 95% CI widths for {label}:")
    for i, name in enumerate(param_names):
        vals = 10**samples[:, i] if "log" in name else samples[:, i]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {name:10}: {hi - lo:.3e}  (95% CI)")

    # Dynamically generate header to match the writing order
    header = ["Label"]
    for name in param_names:
        header.extend([f"{name}_median", f"{name}_CI_lower", f"{name}_CI_upper"])

    # Compute values
    summary_row = [label]
    for i, name in enumerate(param_names):
        vals = samples[:, i]  # ← already in linear space!
        lo, hi = np.percentile(vals, [2.5, 97.5])
        median = np.median(vals)
        summary_row.extend([median, lo, hi])

    # Write row (append mode)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(summary_row)


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

        from mcmc_config import burnin as config_burnin, stride as config_stride
        burnin = args.burnin if args.burnin is not None else config_burnin
        stride = args.stride if args.stride is not None else config_stride

        input_dir = "mcmc_samples"
        summary_path = "posterior_summary.csv"

        # 🧹 Delete old summary if processing all files
        if args.file is None and os.path.exists(summary_path):
            os.remove(summary_path)
            print("🗑️ Removed old posterior_summary.csv")

        def process_file(file_path):
            file_path = file_path.strip()  # ✨ Clean up any extra characters
            label = os.path.basename(file_path).replace("_fitdata.npz", "")
            data = np.load(file_path)
            samples = data["samples"]
            chain = data["chain"]
            t_data = data["t_data"]
            a_data = data["a_data"]
            r = np.max(a_data)
            print(f"📦 Plotting {label} with burnin={burnin}, stride={stride}, r={r:.4f}")
            make_plots(samples, chain, t_data, a_data, r, label,
                       burnin=burnin, stride=stride)

        if args.file:
            args.file = args.file.strip()
            if not os.path.exists(args.file):
                print(f"❌ File not found: {args.file}", flush=True)
                print(f"📂 Raw path: {repr(args.file)}")
            else:
                process_file(args.file)
        else:
            for file_path in load_npz_files(input_dir):
                process_file(file_path)

    except Exception as e:
        print("💥 An error occurred during plotting:", flush=True)
        traceback.print_exc()


