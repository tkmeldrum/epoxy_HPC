"""
BatchBayesian_plots.py

Post-processing for emcee MCMC results saved as *_fitdata.npz files.
Generates posterior overlay, alpha CI, dα/dt, chain, corner, and summary grid plots,
and appends a row to a posterior_summary.csv for downstream Arrhenius analysis.

Usage:
    python BatchBayesian_plots.py                            # process all *_fitdata.npz in mcmc_samples/
    python BatchBayesian_plots.py path/to/file.npz           # process a single file
    python BatchBayesian_plots.py --r 2.0                    # fix stoichiometric ratio (KM model)
    python BatchBayesian_plots.py --input-dir mcmc_samples_nmr --outdir fit_plots_nmr
"""

import os
import csv
import time
import traceback
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import corner
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy import gradient

from mcmc_config import (
    overlay_n as default_overlay_n,
    burnin as default_burnin,
    stride as default_stride,
)


# ---------------------------------------------------------------------------
# ODE model
# ---------------------------------------------------------------------------

def solve_model(log_k1, log_k2, m, n, r, t_data, a_data, eps=1e-10):
    """
    Solve the Kamal-Malkin ODE for a given parameter set.

    The upper clip on alpha is min(1-eps, r-eps) so that (1-a)^(n/2) never
    receives a negative base regardless of r.
    """
    k1 = 10 ** log_k1
    k2 = 10 ** log_k2
    a_upper = min(1.0 - eps, r - eps)
    a0 = [np.clip(a_data[0], eps, a_upper)]

    def ode_rhs(_t, a):
        a_clipped = np.clip(a, eps, a_upper)
        return (k1 + k2 * a_clipped**m) * (1 - a_clipped)**(n / 2) * (r - a_clipped)**(n / 2)

    try:
        sol = solve_ivp(
            ode_rhs,
            [t_data[0], t_data[-1]],
            a0,
            t_eval=t_data,
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            return np.full_like(t_data, np.nan)
        # Clip to physical bounds: alpha must be positive and below r.
        # Small negative excursions near zero are numerical artifacts from LSODA.
        return np.clip(sol.y[0], 1e-10, r - 1e-10)
    except Exception as e:
        print(f"ODE solve failed: {e}")
        return np.full_like(t_data, np.nan)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_posterior_overlay(alpha_preds, t_data, a_data, label, outdir):
    """Plot all posterior curves as a transparent ensemble with the median on top."""
    y_min, y_max = np.min(a_data), np.max(a_data)
    margin = 0.05 * (y_max - y_min)

    median_fit = np.nanmedian(alpha_preds, axis=0)

    plt.figure()
    for a_fit in alpha_preds:
        plt.plot(t_data, a_fit, color="red", alpha=0.1)
    plt.plot(t_data, a_data, "k.", markersize=8, label="Data")
    plt.plot(t_data, median_fit, "r-", lw=2, label="Median fit")
    plt.xlabel("Time")
    plt.ylabel("alpha(t)")
    plt.title(f"{label} — Posterior overlay")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_overlay.png"), dpi=300)
    plt.close()


def plot_alpha_confidence_band(alpha_preds, t_data, a_data, label, outdir):
    """Plot median fit with 95% credible interval band."""
    median_alpha = np.nanmedian(alpha_preds, axis=0)
    lower_alpha  = np.nanpercentile(alpha_preds, 2.5,  axis=0)
    upper_alpha  = np.nanpercentile(alpha_preds, 97.5, axis=0)

    y_min, y_max = np.min(a_data), np.max(a_data)
    margin = 0.05 * (y_max - y_min)

    plt.figure()
    plt.plot(t_data, a_data, "k.", label="Data")
    plt.plot(t_data, median_alpha, "r-", label="Median fit")
    plt.fill_between(t_data, lower_alpha, upper_alpha, color="red", alpha=0.3, label="95% CI")
    plt.xlabel("Time")
    plt.ylabel("alpha(t)")
    plt.title(f"{label} — alpha(t) with 95% CI")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_alpha_ci.png"), dpi=300)
    plt.close()


def plot_dadt_vs_alpha(alpha_preds, t_data, a_data, label, outdir):
    """Plot rate (da/dt) vs conversion (alpha) with posterior CI."""
    dadt_preds   = np.array([gradient(a, t_data) for a in alpha_preds])
    median_dadt  = np.nanmedian(dadt_preds, axis=0)
    lower_dadt   = np.nanpercentile(dadt_preds, 2.5,  axis=0)
    upper_dadt   = np.nanpercentile(dadt_preds, 97.5, axis=0)
    median_alpha = np.nanmedian(alpha_preds, axis=0)

    dadt_data = gradient(a_data, t_data)
    y_min, y_max = np.min(dadt_data), np.max(dadt_data)
    margin = 0.1 * (y_max - y_min)

    plt.figure()
    plt.plot(a_data, dadt_data, "k.", label="Data")
    plt.fill_between(median_alpha, lower_dadt, upper_dadt, color="red", alpha=0.3, label="95% CI")
    plt.plot(median_alpha, median_dadt, "r-", label="Median posterior")
    plt.xlabel("alpha")
    plt.ylabel("dalpha/dt")
    plt.title(f"{label} — dalpha/dt vs alpha")
    plt.ylim(y_min - margin, y_max + margin)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_dadt_alpha.png"), dpi=300)
    plt.close()


def plot_corner(samples, label, outdir):
    """Corner plot of posterior samples, clipping the outermost 1% to reduce outlier distortion."""
    lower_bounds = np.quantile(samples, 0.01, axis=0)
    upper_bounds = np.quantile(samples, 0.99, axis=0)
    mask     = np.all((samples >= lower_bounds) & (samples <= upper_bounds), axis=1)
    filtered = samples[mask]
    ndim     = filtered.shape[1]
    labels   = build_param_names(ndim)
    ranges   = [
        (np.percentile(filtered[:, i], 0.5), np.percentile(filtered[:, i], 99.5))
        for i in range(ndim)
    ]
    fig = corner.corner(filtered, labels=labels, range=ranges)
    fig.savefig(os.path.join(outdir, f"{label}_corner.png"), dpi=300)
    plt.close(fig)


def plot_chains(chain, label, outdir):
    """Trace plot for each parameter across all walkers."""
    n_walkers, n_steps, ndim = chain.shape
    param_names = build_param_names(ndim)
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i in range(ndim):
        for w in range(n_walkers):
            axes[i].plot(chain[w, :, i], alpha=0.3)
        axes[i].set_ylabel(param_names[i])
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} — MCMC chains")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_chains.png"), dpi=300)
    plt.close(fig)


def make_summary_grid(label, outdir):
    """Assemble individual plot images into a single summary figure."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"{label} Summary", fontsize=16)

    def load_and_show(ax, filepath, title=None):
        if os.path.exists(filepath):
            ax.imshow(mpimg.imread(filepath))
            ax.axis("off")
            if title:
                ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)
            ax.axis("off")

    prefix = os.path.join(outdir, label)
    load_and_show(axes[0, 0], f"{prefix}_overlay.png",    "Posterior overlay")
    load_and_show(axes[0, 1], f"{prefix}_alpha_ci.png",   "alpha(t) with 95% CI")
    load_and_show(axes[1, 0], f"{prefix}_dadt_alpha.png", "dalpha/dt vs alpha")
    load_and_show(axes[1, 1], f"{prefix}_chains.png",     "MCMC chains")
    load_and_show(axes[2, 0], f"{prefix}_corner.png",     "Corner plot")

    axes[2, 1].axis("off")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    axes[2, 1].text(
        1.0, -0.1, f"Generated {timestamp}",
        ha="right", va="top", transform=axes[2, 1].transAxes,
        fontsize=10, color="gray",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{prefix}_summary.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def make_plots(samples, chain, t_data, a_data, r, label,
               outdir="fit_plots", summary_path="posterior_summary.csv",
               overlay_n=None, burnin=None, stride=None):
    """
    Full post-processing pipeline for one dataset:
      1. Apply burn-in and thinning.
      2. Remove outlier walkers via Mahalanobis distance.
      3. Draw posterior samples, solve ODE, apply quality filters.
      4. Generate all plots and append a row to posterior_summary.csv.

    Parameters
    ----------
    samples       : flat chain array from emcee (nsamples, ndim)
    chain         : full chain array (nsteps, nwalkers, ndim)
    t_data        : time array for the dataset
    a_data        : observed conversion array for the dataset
    r             : stoichiometric ratio for the ODE (use 2.0 for KM; max(a_data) for DSC)
    label         : string identifier used in filenames and CSV
    outdir        : directory for output plots
    summary_path  : path to the CSV file where the posterior summary row is appended
    """
    burnin   = burnin   if burnin   is not None else default_burnin
    stride   = stride   if stride   is not None else default_stride
    overlay_n = overlay_n if overlay_n is not None else default_overlay_n

    os.makedirs(outdir, exist_ok=True)
    start_time = time.time()
    print(f"Starting plots for {label}")

    # --- Burn-in and thinning ---
    print(f"  samples shape before: {samples.shape}")
    samples = samples[burnin::stride]
    print(f"  samples shape after burnin/stride: {samples.shape}")
    chain = chain[burnin::stride]          # (steps, walkers, ndim)
    chain = np.transpose(chain, (1, 0, 2)) # -> (walkers, steps, ndim)

    # --- Remove outlier walkers by Mahalanobis distance on per-walker means ---
    walker_means = np.mean(chain, axis=1)   # (nwalkers, ndim)

    # Discard walkers with non-finite means before computing covariance.
    finite_mask = np.all(np.isfinite(walker_means), axis=1)
    if not np.all(finite_mask):
        print(f"  Warning: {np.sum(~finite_mask)} walkers have non-finite means and are removed before Mahalanobis filtering.")
        walker_means = walker_means[finite_mask]
        chain        = chain[finite_mask]

    try:
        ensemble_mean = np.mean(walker_means, axis=0)
        ensemble_cov  = np.cov(walker_means.T)
        inv_cov       = np.linalg.pinv(ensemble_cov)
        dists = np.array([
            mahalanobis(walker_means[i], ensemble_mean, inv_cov)
            for i in range(len(walker_means))
        ])
    except np.linalg.LinAlgError as e:
        print(f"  Warning: Mahalanobis distance failed ({e}). Keeping all finite walkers.")
        dists = np.zeros(len(walker_means))

    cutoff    = 3.0
    walker_ok = dists < cutoff
    print(f"  Kept {np.sum(walker_ok)} / {len(walker_ok)} walkers (Mahalanobis cutoff = {cutoff})")
    for i, ok in enumerate(walker_ok):
        if not ok:
            print(f"  Walker {i} removed: distance = {dists[i]:.2f}")

    # Rebuild flat sample array from surviving walkers only
    chain   = chain[walker_ok]
    samples = chain.transpose(1, 0, 2).reshape(-1, chain.shape[-1])

    # Per-walker diagnostic summary
    param_names = build_param_names(chain.shape[2])
    print(f"\n  Per-walker parameter stats (after burnin/stride/walker filtering):")
    for w in range(chain.shape[0]):
        stats = "  ".join(
            f"{name}: {np.mean(chain[w, :, i]):+.4f} +/- {np.std(chain[w, :, i]):.2e}"
            for i, name in enumerate(param_names)
        )
        print(f"    Walker {w}: {stats}")

    # --- Draw posterior curves and apply quality filters ---
    idx    = np.random.choice(len(samples), size=min(overlay_n, len(samples)), replace=False)
    subset = samples[idx]

    alpha_preds     = []
    filtered_subset = []

    for p in subset:
        # Unpack parameters; r may be a free parameter (ndim=6) or fixed externally
        if len(p) == 6:
            log_k1, log_k2, m, n, r_val, log_sigma = p
        else:
            log_k1, log_k2, m, n, log_sigma = p
            r_val = r

        a_fit = solve_model(log_k1, log_k2, m, n, r_val, t_data, a_data)

        if not np.all(np.isfinite(a_fit)):
            continue

        # Filter criteria: curve must stay within plausible bounds and track the data endpoint.
        # Tolerances are relative to the data range to stay scale-independent.
        max_ok      = np.max(a_fit)   < 1.05 * np.max(a_data)
        min_ok      = np.min(a_fit)   > 0
        final_close = np.abs(a_fit[-1] - a_data[-1]) < 0.05 * np.max(a_data)
        monotonic   = np.all(np.diff(a_fit) >= -0.01)

        if not (max_ok and min_ok and final_close and monotonic):
            reasons = []
            if not max_ok:      reasons.append("max_ok")
            if not min_ok:      reasons.append("min_ok")
            if not final_close: reasons.append("final_close")
            if not monotonic:   reasons.append("monotonic")
            print(f"  Curve rejected: {', '.join(reasons)}")
            continue

        alpha_preds.append(a_fit)
        filtered_subset.append(p)

    print(f"  Kept {len(alpha_preds)} / {len(subset)} posterior curves")

    if len(alpha_preds) == 0:
        print(f"  No valid posterior curves for {label}. Saving raw data plot only.")
        plt.figure()
        plt.plot(t_data, a_data, "ko", label="Observed alpha(t)")
        plt.title(f"{label} — Raw data only (no valid posterior curves)")
        plt.xlabel("Time")
        plt.ylabel("alpha(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{label}_raw_data_only.png"), dpi=300)
        plt.close()
        return

    alpha_preds     = np.array(alpha_preds)
    filtered_subset = np.array(filtered_subset)
    print(f"  {len(filtered_subset)}/{len(subset)} curves retained for {label}.")

    # --- Append row to posterior summary CSV ---
    # Done before plots so results are saved even if a plot function crashes.
    # Values are stored in the parameter space used by the sampler (log for rate constants).
    ndim        = filtered_subset.shape[1]
    param_names = build_param_names(ndim)

    print(f"\n  95% CI widths for {label}:")
    for i, name in enumerate(param_names):
        vals   = 10**filtered_subset[:, i] if "log" in name else filtered_subset[:, i]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"    {name:12}: {hi - lo:.3e}")

    header = ["Label"]
    for name in param_names:
        header.extend([f"{name}_median", f"{name}_CI_lower", f"{name}_CI_upper"])

    summary_row = [label]
    for i, name in enumerate(param_names):
        vals   = filtered_subset[:, i]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        summary_row.extend([np.median(vals), lo, hi])

    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(summary_row)
    print(f"  Summary written to {summary_path}")

    # --- Generate plots ---
    t0 = time.time(); plot_posterior_overlay(alpha_preds, t_data, a_data, label, outdir);    print(f"  overlay:       {time.time()-t0:.2f}s")
    t0 = time.time(); plot_alpha_confidence_band(alpha_preds, t_data, a_data, label, outdir); print(f"  alpha CI:      {time.time()-t0:.2f}s")
    t0 = time.time(); plot_dadt_vs_alpha(alpha_preds, t_data, a_data, label, outdir);         print(f"  dα/dt vs α:    {time.time()-t0:.2f}s")
    t0 = time.time(); plot_chains(chain, label, outdir);                                       print(f"  chains:        {time.time()-t0:.2f}s")
    t0 = time.time(); plot_corner(samples, label, outdir);                                     print(f"  corner:        {time.time()-t0:.2f}s")
    t0 = time.time(); make_summary_grid(label, outdir);                                        print(f"  summary grid:  {time.time()-t0:.2f}s")

    print(f"  Finished all plots for {label} in {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------------
# Parameter name lookup
# ---------------------------------------------------------------------------

def build_param_names(ndim):
    """
    Return parameter names matching the column order used by the sampler.
    Base: log_k1, log_k2, m, n
    ndim=5: base + log_sigma        (r fixed)
    ndim=6: base + r + log_sigma    (r free)
    """
    base = ["log_k1", "log_k2", "m", "n"]
    if ndim == 4:
        return base
    if ndim == 5:
        return base + ["log_sigma"]
    if ndim >= 6:
        names = base + ["r", "log_sigma"]
        if ndim > 6:
            names += [f"extra_{i}" for i in range(ndim - 6)]
        return names
    raise ValueError(f"Unexpected ndim: {ndim}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def load_npz_files(directory):
    # Only yield files directly in directory — do not recurse into subdirectories.
    for fname in sorted(os.listdir(directory)):
        full = os.path.join(directory, fname)
        if os.path.isfile(full) and fname.endswith("_fitdata.npz"):
            yield full


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Plot MCMC results from saved *_fitdata.npz files."
        )
        parser.add_argument(
            "file", nargs="?", default=None,
            help="Path to a single .npz file. If omitted, all *_fitdata.npz files in --input-dir are processed.",
        )
        parser.add_argument("--burnin",    type=int,   default=None,          help="Burn-in steps to discard (overrides mcmc_config)")
        parser.add_argument("--stride",    type=int,   default=None,          help="Thinning stride (overrides mcmc_config)")
        parser.add_argument("--r",         type=float, default=None,          help="Fixed stoichiometric ratio r. Defaults to max(a_data) if not set.")
        parser.add_argument("--input-dir", type=str,   default="mcmc_samples", help="Directory to scan for *_fitdata.npz files (default: mcmc_samples)")
        parser.add_argument("--outdir",    type=str,   default="fit_plots",   help="Output directory for plots (default: fit_plots)")
        parser.add_argument("--summary",   type=str,   default=None, help="Path for the posterior summary CSV. Defaults to posterior_summary_{timestamp}.csv")
        args = parser.parse_args()

        if args.summary is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            args.summary = f"posterior_summary_{timestamp}.csv"

        from mcmc_config import burnin as config_burnin, stride as config_stride
        burnin = args.burnin if args.burnin is not None else config_burnin
        stride = args.stride if args.stride is not None else config_stride

        def process_file(file_path):
            file_path = file_path.strip()
            label  = os.path.basename(file_path).replace("_fitdata.npz", "")
            data   = np.load(file_path)
            t_data = data["t_data"]
            a_data = data["a_data"]
            r      = args.r if args.r is not None else np.max(a_data)
            print(f"Plotting {label}  burnin={burnin}  stride={stride}  r={r:.4f}")
            make_plots(
                data["samples"], data["chain"], t_data, a_data, r, label,
                outdir=args.outdir,
                summary_path=args.summary,
                burnin=burnin,
                stride=stride,
            )

        def process_file_safe(file_path):
            try:
                process_file(file_path)
            except Exception:
                print(f"Failed to process {file_path}:")
                traceback.print_exc()
                print("Continuing with remaining files.")

        if args.file:
            file_path = args.file.strip()
            if not os.path.exists(file_path):
                print(f"File not found: {file_path!r}")
            else:
                process_file_safe(file_path)
        else:
            for file_path in load_npz_files(args.input_dir):
                process_file_safe(file_path)

    except Exception:
        print("An error occurred during setup:")
        traceback.print_exc()
