import pandas as pd
import numpy as np
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

# === Helper ===
def parse_label(label):
    parts = label.split('_')
    return parts[0], parts[1], int(parts[2][:-1])  # e.g., DSC_EDA_50C

# === Plot 1: Parameter trends vs 1/T ===
def plot_extended_arrhenius(df, x_col='1/T [K-1]',
                             ylim_k=(-10, 0), ylim_m=(0, 2.5), ylim_n=(0, 2.5), ylim_r=(0, 1)):
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, sample in enumerate(samples):
        for method in methods:
            subset = df[(df['Sample'] == sample) & (df['Method'] == method)]
            if not subset.empty:
                x = subset[x_col]

                # ln(k1)
                y1 = subset['Fit_k1']
                yerr1 = subset['Unc_k1']
                axes[i, 0].errorbar(x, y1, yerr=yerr1, fmt='o', color=colors[method], capsize=3, label=method)
                axes[i, 0].set_ylim(*ylim_k)
                axes[i, 0].set_title(f'{sample} - ln(k₁)')

                # ln(k2)
                y2 = subset['Fit_k2']
                yerr2 = subset['Unc_k2']
                axes[i, 1].errorbar(x, y2, yerr=yerr2, fmt='o', color=colors[method], capsize=3)
                axes[i, 1].set_ylim(*ylim_k)
                axes[i, 1].set_title(f'{sample} - ln(k₂)')

                # m
                y3 = subset['Fit_m']
                yerr3 = subset['Unc_m']
                axes[i, 2].errorbar(x, y3, yerr=yerr3, fmt='o', color=colors[method], capsize=3)
                axes[i, 2].set_ylim(*ylim_m)
                axes[i, 2].set_title(f'{sample} - m')

                # n
                y4 = subset['Fit_n']
                yerr4 = subset['Unc_n']
                axes[i, 3].errorbar(x, y4, yerr=yerr4, fmt='o', color=colors[method], capsize=3)
                axes[i, 3].set_ylim(*ylim_n)
                axes[i, 3].set_title(f'{sample} - n')

                # # r
                # y5 = subset['Fit_r']
                # yerr5 = subset['Unc_r']
                # axes[i, 4].errorbar(x, y5, yerr=yerr5, fmt='o', color=colors[method], capsize=3)
                # axes[i, 4].set_ylim(*ylim_r)
                # axes[i, 4].set_title(f'{sample} - r')

            if i == 2:
                for j in range(4):
                    axes[i, j].set_xlabel('1/T [K$^{-1}$]')
            if method == 'DSC':
                axes[i, 0].set_ylabel('ln(k₁)')
                axes[i, 1].set_ylabel('ln(k₂)')
                axes[i, 2].set_ylabel('m')
                axes[i, 3].set_ylabel('n')
                # axes[i, 4].set_ylabel('r')

    axes[0, 0].legend(title='Method')
    axes[0, 1].legend(title='Method')
    plt.suptitle('Parameter Trends vs 1/T (K$^{-1}$)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# === Plot 2: ln(k) Arrhenius fits ===
def plot_arrhenius_fits(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, sample in enumerate(samples):
        for j, (fit_col, unc_col, title) in enumerate([('Fit_k1', 'Unc_k1', 'ln(k₁)'), ('Fit_k2', 'Unc_k2', 'ln(k₂)')]):
            ax = axes[i, j]
            for method in methods:
                subset = df[(df['Sample'] == sample) & (df['Method'] == method)]
                if subset.empty:
                    continue
                x = subset['1/T [K-1]'].values
                y = subset[fit_col].values*np.log(10)
                yerr = subset[unc_col].values *np.log(10)

                popt, pcov = curve_fit(lambda x, m, b: m * x + b, x, y, sigma=yerr, absolute_sigma=True)
                slope, intercept = popt
                perr = np.sqrt(np.diag(pcov))

                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = slope * x_fit + intercept
                ax.errorbar(x, y, yerr=yerr, fmt='o', color=colors[method], capsize=3, label=f'{method} data')
                ax.plot(x_fit, y_fit, '--', color=colors[method], label=f'{method} fit')

                Ea = -slope * R / 1000
                Ea_err = perr[0] * R / 1000
                ax.text(0.65, 0.1 - 0.12 * methods.index(method),
                        f"{method}: Ea={Ea:.1f}±{Ea_err:.1f} kJ/mol",
                        transform=ax.transAxes, fontsize=8, color=colors[method])

            ax.set_title(f'{sample} - {title}')
            ax.set_ylim(-20, -2)
            if i == 2:
                ax.set_xlabel('1/T [K⁻¹]')
            if j == 0:
                ax.set_ylabel(f'{title}')
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
    df['Fit_r'] = 1.0
    df['Unc_r'] = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig1 = plot_extended_arrhenius(df)
    fig1.savefig(f"fit_trends_{timestamp}.pdf")

    fig2 = plot_arrhenius_fits(df)
    fig2.savefig(f"arrhenius_fits_{timestamp}.pdf")