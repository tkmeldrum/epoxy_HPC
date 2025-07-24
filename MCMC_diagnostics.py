import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for saving plots to files

import matplotlib.pyplot as plt
import argparse
import emcee
import warnings
import pickle
import time
import arviz as az
from model_module import log_posterior
from emcee.autocorr import AutocorrError

from mcmc_config import burnin, stride, overlay_n, nwalkers, nsteps

param_names = ["log_k1", "log_k2", "m", "n", "log_sigma"]

csv_path = "diagnostic_plots/diagnostics_summary.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w") as f:
        f.write("label,parameter,tau,neff,rhat\n")

def load_chain_data(npz_path):
    npz_path = npz_path.strip()  # ✨ THIS LINE REMOVES \r, \n, spaces
    data = np.load(npz_path)
    samples = data["samples"]
    chain = data["chain"]

    # Load corresponding sampler pickle
    pkl_path = npz_path.replace("_fitdata.npz", "_sampler.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            sampler = pickle.load(f)
    else:
        sampler = None

    return samples, chain, sampler

def compute_autocorr_stats(sampler):
    try:
        tau = sampler.get_autocorr_time(tol=0)
        if np.any(np.isnan(tau)) or np.any(tau <= 0):
            raise ValueError("Invalid autocorrelation time (NaN or nonpositive).")
        ess = sampler.get_chain().shape[0] * sampler.get_chain().shape[1] / tau
        return tau, ess
    except Exception as e:
        print(f"⚠️ Autocorr time estimation failed: {e}")
        return None, None



def plot_diagnostics(chain, label, param_names=None, outdir="diagnostic_plots", burnin=burnin):
    os.makedirs(outdir, exist_ok=True)

    # Handle correct shape: (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    print(f"🔍 Chain shape for plotting: steps={nsteps}, walkers={nwalkers}, params={ndim}")

    if param_names is None:
        param_names = [f"param_{i}" for i in range(ndim)]

    plot_stride = 10
    max_walkers_to_plot = min(20, nwalkers)

    # Trace plots
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    for i in range(ndim):
        for walker in range(max_walkers_to_plot):
            axes[i].plot(
                np.arange(0, nsteps, plot_stride),
                chain[::plot_stride, walker, i],
                alpha=0.3
            )
        axes[i].set_ylabel(param_names[i])
        axes[i].axvspan(0, burnin, color='red', alpha=0.1, label="Burn-in" if i == 0 else None)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} Trace Plots")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_trace.png")
    plt.close(fig)

    # Running mean plots
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    for i in range(ndim):
        running_mean = np.mean(chain[::plot_stride, :, i], axis=1)
        axes[i].plot(np.arange(0, nsteps, plot_stride), running_mean, label=f"Mean of {param_names[i]}")
        axes[i].set_ylabel(param_names[i])
        axes[i].axvspan(0, burnin, color='red', alpha=0.1, label="Burn-in" if i == 0 else None)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} Running Means")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_running_means.png")
    plt.close(fig)

def summarize_diagnostics(samples, chain, label, sampler, burnin, csv_path=None):
    print(f"\n📊 Diagnostics for {label}")
    print(f"Shape of chain: {chain.shape} (steps, walkers, params)")
    print(f"🧹 Using burn-in of {burnin} steps")

    # Step 1: Autocorrelation diagnostics
    if chain.shape[0] >= 1000:
        tau, ess = compute_autocorr_stats(sampler)
        if tau is not None:
            for i, t in enumerate(tau):
                print(f"Param {i}: τ = {t:.1f}, Neff = {ess[i]:.0f}")
            print(f"📏 Recommendation: run length ≥ {int(50 * np.max(tau))} steps")
        else:
            print("⚠️ Could not estimate autocorrelation time")
    else:
        print("ℹ️ Chain too short to reliably estimate autocorrelation time")
        tau, ess = None, None

    # Step 2: ArviZ conversion and R̂
    try:
        nsteps, nwalkers, ndim = chain.shape
        print(f"🔍 Interpreting as {nwalkers} walkers over {nsteps} steps, {ndim} parameters")
        thin_chain = chain[burnin::stride, :, :]  # (steps, walkers, params)
        print(f"📏 Thin chain shape: {thin_chain.shape} (steps, walkers, params)")

        posterior = {}
        for i, name in enumerate(param_names):
            arr = thin_chain[:, :, i].T  # (chains, draws)
            print(f"  ↪️ {name}: shape {arr.shape} (chains, draws)")
            posterior[name] = arr

        idata_start = time.time()
        idata = az.from_dict(posterior=posterior)
        idata_end = time.time()
        print("✅ Created ArviZ InferenceData")

        rhat_start = time.time()
        rhat = az.rhat(idata)
        rhat_end = time.time()

        print("R̂ (Gelman-Rubin):")
        for var in rhat.data_vars:
            print(f"  {var}: {rhat[var].values.item():.6f}")

        print(f"⏱️ az.from_dict() took {idata_end - idata_start:.2f}s")
        print(f"⏱️ az.rhat() took {rhat_end - rhat_start:.2f}s")

        # Step 3: CSV output (optional)
        if csv_path:
            for i, name in enumerate(param_names):
                tau_val = tau[i] if tau is not None else np.nan
                neff_val = ess[i] if ess is not None else np.nan
                rhat_val = rhat[name].values.item()
                with open(csv_path, "a") as f:
                    f.write(f"{label},{name},{tau_val:.4f},{neff_val:.1f},{rhat_val:.6f}\n")

    except Exception as e:
        print(f"⚠️ R̂ computation or CSV export failed: {e}")

    return tau, ess





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default=None,
                        help="Optional path to a single .npz file. If omitted, all *_fitdata.npz files in mcmc_samples/ will be processed.")
    args = parser.parse_args()

    input_dir = "mcmc_samples"
    files = [args.file] if args.file else [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith("_fitdata.npz")
    ]

    print(f"📁 Found {len(files)} file(s) to process.")

    for idx, file in enumerate(files, 1):
        label = os.path.basename(file).replace("_fitdata.npz", "")
        print(f"\n🔍 [{idx}/{len(files)}] Processing: {label}")

        t0 = time.time()
        samples, chain, sampler = load_chain_data(file)
        print(f"📥 Loaded data in {time.time() - t0:.2f}s")

        # === Diagnostics on full chain ===
        t1 = time.time()
        summarize_diagnostics(samples=None, chain=chain, label=label, sampler=sampler, burnin=burnin, csv_path=csv_path)
        print(f"✅ Summarized diagnostics in {time.time() - t1:.2f}s")

        # === Thinning for ArviZ and plotting ===
        thinned = chain[burnin::stride, :, :]  # (thinned_steps, walkers, params)
        n_steps, n_walkers, n_params = thinned.shape
        print(f"📏 Thinned chain shape: {thinned.shape} (steps, walkers, params)")

        # === Reshape for ArviZ ===
        posterior_dict = {
            name: thinned[:, :, i].T  # (chains, draws)
            for i, name in enumerate(param_names)
        }
        for name in param_names:
            print(f"  ↪️ {name}: shape {posterior_dict[name].shape} (chains, draws)")

        idata = az.from_dict(posterior=posterior_dict)
        print("✅ Created ArviZ InferenceData")

        # === Gelman-Rubin R̂ ===
        t2 = time.time()
        rhat_vals = az.rhat(idata)
        print("R̂ (Gelman-Rubin):")
        for name in param_names:
            rhat_val = rhat_vals[name].values.item()
            print(f"  {name}: {rhat_val:.6f}")
        print(f"⏱️ az.rhat() took {time.time() - t2:.2f}s")

        # === Diagnostic plots ===
        t3 = time.time()
        plot_diagnostics(thinned, label, param_names=param_names)
        print(f"📊 Plots saved to 'diagnostic_plots/' in {time.time() - t3:.2f}s")

        total_time = time.time() - t0
        print(f"⏱️ Total file time: {total_time:.2f}s")



