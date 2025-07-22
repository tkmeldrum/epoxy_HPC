import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import emcee
import corner
import argparse
import pickle
from numpy import gradient
import multiprocessing
from multiprocessing import freeze_support
from mcmc_config import burnin, stride, overlay_n, nwalkers, nsteps


# param_names = ['log_k1', 'log_k2', 'm', 'n', 'r', 'log_sigma']
param_names = ['log_k1', 'log_k2', 'm', 'n', 'log_sigma']
samples_list = ['EDA', 'DAP', 'DAB']
dsc_temps = [25, 33, 50, 60, 80, 100]
nmr_temps = [25, 33, 40]

# Data loading
mat = loadmat('epoxy_data.mat')

try:
    import pandas as pd
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
            return np.full_like(t_data, np.nan)
        return sol.y[0]
    except Exception as e:
        print(f"ODE solver failed with exception: {e}")
        return np.full_like(t_data, np.nan)

def log_prior(params, t_data, a_data, r):
    log_k1, log_k2, m, n,  log_sigma = params
    if not (-8 < log_k1 < -1 and
            -8 < log_k2 < -1 and
             0 < m < 3 and
             0 < n < 3 and
            #  0.9 * np.max(a_data) < r < 1.1 * np.max(a_data) and
            -12 < log_sigma < 0):
        return -np.inf
    return 0.0

def log_likelihood(params, t_data, a_data, r):
    log_k1, log_k2, m, n, log_sigma = params
    # print(f"Inside log_likelihood: {log_k1=}, {log_k2=}, {m=}, {n=}, {log_sigma=}, {r=}")
    if r <= 0 or not np.isfinite(r):
        return -np.inf  # 🚫 invalid r

    try:
        a_fit = solve_model(log_k1, log_k2, m, n, r, t_data, a_data)
    except Exception:
        return -np.inf  # catch rare integration failures

    if (
        not np.all(np.isfinite(a_fit)) or
        np.any(a_fit <= 0) or
        np.any(a_fit >= r)
    ):
        return -np.inf  # strict penalty: don't reward broken curves

    sigma = 10 ** log_sigma
    residual = (a_data - a_fit) / sigma
    return -0.5 * np.sum(residual**2 + np.log(2 * np.pi * sigma**2))


def log_posterior(params, t_data, a_data, r):
    lp = log_prior(params, t_data, a_data, r)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, t_data, a_data, r)
    if not np.isfinite(ll):
        print(f"[Reject] {params=}")
        return -np.inf
    return lp + ll


def run_mcmc(start_params, t_data, a_data, r, scale_params=None):
    ndim = len(start_params)
    if scale_params is None:
        scale_params = [1e-4] * ndim

    p0 = np.array([
        start_params + scale_params * np.random.randn(ndim)
        for _ in range(nwalkers)
    ])

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

    if method == 'NMR':
        dataset_name = 'NMR'
        offset = {'EDA': 0, 'DAP': 3, 'DAB': 6}[sample]
        ii = offset + [25, 33, 40].index(temp)
    else:
        dataset_name = sample
        ii = [25, 33, 50, 60, 80, 100].index(temp)

    try:
        t_data = np.squeeze(mat[dataset_name][0, 0]['clean_time'][0, ii])
        a_data = np.squeeze(mat[dataset_name][0, 0]['clean_alpha_unscaled'][0, ii])
        t_data = t_data - t_data[0]
        a_data = np.clip(a_data, 1e-6, 1 - 1e-6)

        imax = np.argmax(a_data)
        t_data = t_data[:imax + 1]
        a_data = a_data[:imax + 1]
    except Exception:
        print(f"Dataset {method}_{sample}_{temp}C not found.")
        return None

    if fit_csv is not None:
        row = fit_csv[
            (fit_csv["Method"] == method) &
            (fit_csv["Sample"] == sample) &
            (fit_csv["Temperature"] == temp)
        ]
        if not row.empty:
            row = row.iloc[0]
            log_k1 = np.log10(row["Fit_k1"])
            log_k2 = np.log10(row["Fit_k2"])
            m = row["Fit_m"]
            n = row["Fit_n"]
            # r = row["Fit_r"]
            r = np.max(a_data)  # fix r based on data
            sigma_guess = np.std(a_data - np.clip(a_data, 1e-6, 1 - 1e-6))
            log_sigma = np.log10(sigma_guess) if sigma_guess > 0 else -4
            start = [log_k1, log_k2, m, n, log_sigma]

            try:
                scale_params = np.array([
                    row["Unc_k1"] / row["Fit_k1"] / np.log(10) if row["Fit_k1"] > 0 else 0.2,
                    row["Unc_k2"] / row["Fit_k2"] / np.log(10) if row["Fit_k2"] > 0 else 0.2,
                    row["Unc_m"],
                    row["Unc_n"],
                    # row["Unc_r"],
                    0.2  # default sigma unc
                ])
            except Exception as e:
                print(f"[Warning] Missing uncertainty values for {method}_{sample}_{temp}C: {e}")
                scale_params = np.array([0.2] * 5)

            # 🛡 Apply minimum spread floor to prevent stuck walkers
            scale_params = np.maximum(scale_params, 0.05)

        else:
            print(f"[Fallback] No CSV match for {method}_{sample}_{temp}C. Using default start.")
            start = [-5, -2, 0.5, 1.4, -4]
            scale_params = np.array([0.2] * 5)
    else:
        start = [-5, -2, 0.5, 1.4, -4]
        scale_params = np.array([0.2] * 5)

    print(f"🔧 MCMC init: start = {start}")
    print(f"🔧 MCMC init: scale_params = {scale_params}")


    # Attempt optimization
    opt = minimize(lambda p: -log_posterior(p, t_data, a_data, r), start,
                   method='Nelder-Mead', options={'maxiter': 1000})

    if not opt.success or np.any(np.isnan(opt.x)):
        print(f"[Fallback] Optimization failed for {sample}, {method}, {temp}. Proceeding with CSV/default start.")
        init_params = start
    else:
        init_params = opt.x


    print("Start params:", start)
    print("r (fixed):", r)


    # Run MCMC with either optimized or fallback
    samples, chain, sampler = run_mcmc(init_params, t_data, a_data, r, scale_params)

    # Save outputs
    label = f"{method}_{sample}_{temp}C"
    os.makedirs("mcmc_samples", exist_ok=True)

    np.savez(f"mcmc_samples/{label}_fitdata.npz",
             samples=samples,
             chain=chain,
             t_data=t_data,
             a_data=a_data,
             log_prob=sampler.get_log_prob())

    with open(f"mcmc_samples/{label}_sampler.pkl", "wb") as f:
        pickle.dump(sampler, f)

    # Summarize
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
    args = parser.parse_args()

    output_dir = "fit_results"
    os.makedirs(output_dir, exist_ok=True)

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