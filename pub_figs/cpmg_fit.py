"""CPMG stretched-exponential fit parameters: T₂ and β vs elapsed time.

Shows how the NMR relaxation time (T₂) and stretching exponent (β) evolve
as the epoxy cures — the underlying observables from which α is derived.

Default: EDA 25 °C  (change SAMPLE / TEMP_STR to select another dataset)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pub_utils as pu

SAMPLE   = 'EDA'
TEMP_STR = '25C'

T2_LIM   = (1e-5, 1e-1)   # log-scale T₂ axis (s)
BETA_LIM = (0.0,  2.5)    # linear β axis


def _cap_err(vals, errs, max_relative=0.2):
    """Cap error bars at max_relative * |value| to keep plots legible."""
    return np.minimum(errs, max_relative * np.abs(vals))


def _asterisks(ax, t_all, y_all, ylo, yhi, color='k'):
    """Plot ▲/▼ asterisk markers at axis boundaries for out-of-range points."""
    hi = y_all > yhi
    lo = y_all < ylo
    if hi.any():
        ax.scatter(t_all[hi], np.full(hi.sum(), yhi * 0.97),
                   marker='*', color=color, s=50, zorder=5, clip_on=False)
    if lo.any():
        ax.scatter(t_all[lo], np.full(lo.sum(), ylo * 1.05),
                   marker='*', color=color, s=50, zorder=5, clip_on=False)


def make_cpmg_figure(sample, temp_str, stem=None):
    """Generate and save CPMG T₂/β figure for one dataset.

    stem: path relative to pub_utils._FIGURES (without extension).
          Defaults to 'SI_figures/cpmg_fit_{sample}_{temp_str}'.
    Good fits: filled black circles. Dropped scans: faded red markers.
    Out-of-range points: asterisk at the axis boundary.
    Gray shading marks the temporal co-adding region (n_avg > 1).
    """
    if stem is None:
        stem = f'SI_figures/cpmg_fit_{sample}_{temp_str}'

    df       = pd.read_csv(pu.NMR_RAW)
    all_mask = (df['sample'] == sample) & (df['temp'] == temp_str)
    sub      = df[all_mask].dropna(subset=['T2', 'beta']).sort_values('elapsed_min')

    good    = sub[~sub['dropped']] if 'dropped' in sub.columns else sub
    dropped = sub[sub['dropped']]  if 'dropped' in sub.columns else sub.iloc[0:0]

    t    = good['elapsed_min'].to_numpy()
    T2   = good['T2'].to_numpy()
    T2e  = _cap_err(good['T2'], good['T2_err'])
    beta = good['beta'].to_numpy()
    be   = _cap_err(good['beta'], good['beta_err'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5.0), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    ax1.errorbar(t, T2, yerr=T2e,
                 fmt='o', ms=4, color='k', mfc='k',
                 capsize=3, lw=0.8, elinewidth=0.8)
    ax1.set_yscale('log')
    ax1.set_ylim(*T2_LIM)
    ax1.set_ylabel(r'$T_2$ (s)')

    ax2.errorbar(t, beta, yerr=be,
                 fmt='o', ms=4, color='k', mfc='k',
                 capsize=3, lw=0.8, elinewidth=0.8)
    ax2.set_ylim(*BETA_LIM)
    ax2.set_ylabel(r'$\beta$')
    ax2.set_xlabel('Elapsed time (min)')

    # Dropped scans as faded red markers (clipped to axis range)
    if not dropped.empty:
        ax1.scatter(dropped['elapsed_min'],
                    dropped['T2'].clip(*T2_LIM),
                    color='red', alpha=0.4, zorder=3, s=20)
        ax2.scatter(dropped['elapsed_min'],
                    dropped['beta'].clip(*BETA_LIM),
                    color='red', alpha=0.4, zorder=3, s=20)

    # Asterisks for good points outside axis limits
    _asterisks(ax1, t, T2,   *T2_LIM)
    _asterisks(ax2, t, beta, *BETA_LIM)

    # Shade temporal co-adding region
    if 'n_avg' in sub.columns and (sub['n_avg'] > 1).any():
        avg_rows = sub[sub['n_avg'] > 1]
        t_shade  = avg_rows['elapsed_min'].min()
        t_end    = sub['elapsed_min'].max()
        for ax in (ax1, ax2):
            ax.axvspan(t_shade, t_end, alpha=0.08, color='gray', zorder=0)

    pu.savefig(fig, stem)
    plt.close(fig)
    return stem


def main():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    })
    pu.snapshot_data()
    make_cpmg_figure(SAMPLE, TEMP_STR, stem='cpmg_fit')
    pu.write_provenance(
        __file__,
        datasets=[f'NMR raw CPMG fit results — sample: {SAMPLE}, temp: {TEMP_STR} (T₂ and β vs elapsed time)'],
        source_paths=[pu.NMR_RAW],
    )


if __name__ == '__main__':
    main()
