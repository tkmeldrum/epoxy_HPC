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

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

SAMPLE   = 'EDA'
TEMP_STR = '25C'


def main():
    pu.snapshot_data()
    df = pd.read_csv(pu.NMR_RAW)

    mask = ((df['sample'] == SAMPLE) &
            (df['temp']   == TEMP_STR) &
            (~df['dropped']))
    sub  = df[mask].dropna(subset=['T2', 'beta']).sort_values('elapsed_min')

    t    = sub['elapsed_min'].to_numpy()
    T2   = sub['T2'].to_numpy()
    T2e  = sub['T2_err'].to_numpy()
    beta = sub['beta'].to_numpy()
    be   = sub['beta_err'].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5.0), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # ── top: T₂ (log scale) ───────────────────────────────────────────────
    ax1.errorbar(t, T2, yerr=T2e,
                 fmt='o', ms=4, color='k', mfc='k',
                 capsize=3, lw=0.8, elinewidth=0.8)
    ax1.set_ylabel(r'$T_2$ (s)')

    # ── bottom: β ─────────────────────────────────────────────────────────
    ax2.errorbar(t, beta, yerr=be,
                 fmt='o', ms=4, color='k', mfc='k',
                 capsize=3, lw=0.8, elinewidth=0.8)
    ax2.set_ylabel(r'$\beta$')
    ax2.set_xlabel('Elapsed time (min)')

    pu.savefig(fig, 'cpmg_fit')
    pu.write_provenance(
        __file__,
        datasets=[f'NMR raw CPMG fit results — sample: {SAMPLE}, temp: {TEMP_STR} (T₂ and β vs elapsed time)'],
        source_paths=[pu.NMR_RAW],
    )
    plt.close(fig)


if __name__ == '__main__':
    main()
