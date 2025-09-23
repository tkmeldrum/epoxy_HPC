import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# === Config ===
R = 8.3145  # J/mol·K
samples = ['EDA', 'DAP', 'DAB']
methods = ['DSC', 'NMR']
colors = {'DSC': 'red', 'NMR': 'blue'}

# === Helpers ===
# Parameters we care about, in order
ALL_PARAMS = ["k1", "k2", "m", "n", "r"]  # r is optional

# Nice labels for plotting (edit as you prefer)
PLOT_LABEL = {
    "k1": "ln(k₁)",
    "k2": "ln(k₂)",
    "m": "m",
    "n": "n",
    "r": "r",
}

# def present_params(df):
#     """
#     Detect which parameters actually exist in the input dataframe.
#     Looks for columns like '<p>_median' (e.g., 'r_median').
#     Order is preserved (k1,k2,m,n[,r]).
#     """
#     present = []
#     for p in ALL_PARAMS:
#         # accept either '<p>_median' or 'log_<p>_median'
#         if f"{p}_median" in df.columns or f"log_{p}_median" in df.columns:
#             present.append(p)
#     return present

def ensure_fit_unc_columns(df):
    def add_param(p, is_log=False):
        src = f"log_{p}" if is_log else p
        df[f"Fit_{p}"] = df[f"{src}_median"]
        df[f"Unc_{p}"] = 0.5 * (df[f"{src}_CI_upper"] - df[f"{src}_CI_lower"])

    # k1/k2: prefer your existing creation, otherwise fill here
    for p in ("k1", "k2"):
        if f"Fit_{p}" not in df:
            if f"log_{p}_median" in df.columns:
                add_param(p, is_log=True)
            elif f"{p}_median" in df.columns:
                add_param(p, is_log=False)

    # m, n
    for p in ("m", "n"):
        if f"Fit_{p}" not in df and f"{p}_median" in df.columns:
            add_param(p, is_log=False)

    # r (optional)
    if f"Fit_r" not in df and "r_median" in df.columns:
        add_param("r", is_log=False)

def parse_label(label):
    # e.g., "DSC_EDA_50C" → ("DSC","EDA",50)
    m = re.match(r"^([A-Za-z]+)_([A-Za-z]+)_(\d+)[cC]$", label.strip())
    if not m:
        raise ValueError(f"Unrecognized label format: {label}")
    method, sample, tempC = m.group(1), m.group(2), int(m.group(3))
    return method, sample, tempC

# === Plot 1: Parameter trends vs 1/T ===
def plot_extended_arrhenius(df, x_col='1/T [K-1]',
                            ylim_k=(-10, 0), ylim_m=(0, 2.5), ylim_n=(0, 2.5), ylim_r=(0, 1)):

    params = [p for p in ("k1","k2","m","n","r") if f"Fit_{p}" in df.columns]
    if not params:
        fig = plt.figure(figsize=(4,3))
        plt.text(0.5, 0.5, "No parameters to plot", ha='center', va='center')
        plt.axis("off")
        return fig
    ncols = len(params)
    fig, axes = plt.subplots(len(samples), ncols, figsize=(5*ncols, 3.2*len(samples)), sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    def _ylim_for(p):
        if p in ("k1","k2"): return ylim_k
        if p == "m": return ylim_m
        if p == "n": return ylim_n
        if p == "r": return ylim_r
        return None

    for i, sample in enumerate(samples):
        for method in methods:
            subset = df[(df['Sample'] == sample) & (df['Method'] == method)]
            if subset.empty:
                continue
            x = subset[x_col].values

            for j, p in enumerate(params):
                y    = subset[f"Fit_{p}"].values
                yerr = subset[f"Unc_{p}"].values
                
                # convert log10(k) → ln(k) for k1/k2 so labels stay correct
                if p in ("k1", "k2"):
                    y    = y * np.log(10.0)
                    yerr = yerr * np.log(10.0)

                axes[i, j].errorbar(x, y, yerr=yerr, fmt='o', color=colors[method], capsize=3, label=method)
                yl = _ylim_for(p)
                if yl is not None:
                    axes[i, j].set_ylim(*yl)
                axes[i, j].set_title(f'{sample} - {PLOT_LABEL[p]}')

        # x-labels on last row
        for j, p in enumerate(params):
            if i == len(samples) - 1:
                axes[i, j].set_xlabel('1/T [K$^{-1}$]')
            # y-labels on first method only (DSC gate) – keep your style
            if "DSC" in methods and j < ncols:
                axes[i, j].set_ylabel(PLOT_LABEL[params[j]])

    # One legend per row (on last axis in the row)
    for i in range(len(samples)):
        axes[i, -1].legend(title='Method', fontsize=8, frameon=False)

    plt.suptitle('Parameter Trends vs 1/T (K$^{-1}$)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# === Plot 2: ln(k) Arrhenius fits ===
def plot_arrhenius_fits(df):
    k_cols = [(p, f"Fit_{p}", f"Unc_{p}") for p in ("k1","k2") if f"Fit_{p}" in df.columns]
    if not k_cols:
        # Nothing to plot
        fig = plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "No k-parameters found", ha='center', va='center')
        plt.axis("off")
        return fig

    ncols = len(k_cols)
    fig, axes = plt.subplots(len(samples), ncols, figsize=(5*ncols, 3.8*len(samples)), sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, sample in enumerate(samples):
        for j, (p, fit_col, unc_col) in enumerate(k_cols):
            ax = axes[i, j]
            for method in methods:
                subset = df[(df['Sample'] == sample) & (df['Method'] == method)]
                if subset.empty:
                    continue
                x = subset['1/T [K-1]'].values
                y = subset[fit_col].values * np.log(10.0)
                yerr = subset[unc_col].values * np.log(10.0)

                if x.size < 2:
                    # Not enough points to fit; still plot the points/errbars
                    ax.errorbar(x, y, yerr=yerr, fmt='o', color=colors[method], capsize=3, label=f'{method} data')
                    continue

                # Avoid zero sigmas to keep curve_fit happy
                if np.any(yerr <= 0) or np.allclose(yerr, 0):
                    yerr = None
                    popt, pcov = curve_fit(lambda X, m, b: m*X + b, x, y)
                    perr = np.sqrt(np.diag(pcov))
                else:
                    popt, pcov = curve_fit(lambda X, m, b: m*X + b, x, y, sigma=yerr, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))

                slope, intercept = popt
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept

                ax.errorbar(x, y, yerr=yerr, fmt='o', color=colors[method], capsize=3, label=f'{method} data')
                ax.plot(x_fit, y_fit, '--', color=colors[method], label=f'{method} fit')

                Ea     = -slope * R / 1000.0
                Ea_err =  perr[0] * R / 1000.0 if perr.size>0 else np.nan
                ax.text(0.62, 0.12 - 0.12 * methods.index(method),
                        f"{method}: Ea={Ea:.1f}±{Ea_err:.1f} kJ/mol",
                        transform=ax.transAxes, fontsize=8, color=colors[method])

            ax.set_title(f'{sample} – ln(k{1 if p=="k1" else 2})')
            ax.set_ylim(-20, -2)
            if i == len(samples) - 1:
                ax.set_xlabel('1/T [K⁻¹]')
            if j == 0:
                ax.set_ylabel('ln(k)')
            ax.legend(fontsize=8)
            ax.grid(True)

    plt.suptitle('Arrhenius Plots with Linear Fits (ln scale)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# === MAIN ===
if __name__ == "__main__":
    df = pd.read_csv("posterior_summary.csv")
    df[['Method', 'Sample', 'Temp']] = df['Label'].apply(lambda x: pd.Series(parse_label(x)))
    df['1/T [K-1]'] = 1 / (df['Temp']+273.15)

    rename_dict = {'log_k1': 'k1', 'log_k2': 'k2', 'log_sigma': 'sigma'}
    for old, new in rename_dict.items():
        df[f'Fit_{new}'] = df[f'{old}_median']
        df[f'Unc_{new}'] = np.abs(df[f'{old}_CI_upper'] - df[f'{old}_CI_lower']) / 2

    df['Fit_m'] = df['m_median']
    df['Unc_m'] = np.abs(df['m_CI_upper'] - df['m_CI_lower']) / 2
    df['Fit_n'] = df['n_median']
    df['Unc_n'] = np.abs(df['n_CI_upper'] - df['n_CI_lower']) / 2
    ensure_fit_unc_columns(df)
    # df['Fit_r'] = 1.0
    # df['Unc_r'] = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig1 = plot_extended_arrhenius(df)
    fig1.savefig(f"fit_trends_{timestamp}.pdf")

    fig2 = plot_arrhenius_fits(df)
    fig2.savefig(f"arrhenius_fits_{timestamp}.pdf")