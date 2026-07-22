"""Arrhenius plots: 3×2 grid (rows = EDA/DAP/DAB, cols = ln k1 / ln k2)."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import pub_utils as pu

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

SAMPLES = ['EDA', 'DAP', 'DAB']  # row order top to bottom
PARAMS  = [('lnk1', r'$k_1$', (-25, 0)),
           ('lnk2', r'$k_2$', (-12.5, 0))]

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()

    # NMR2 folded into NMR for fit lines
    df_fit = df.copy()
    df_fit.loc[df_fit['Method'] == 'NMR2', 'Method'] = 'NMR'

    # Global x-range with padding
    x_all = df['inv_T'].to_numpy()
    x_pad = (x_all.max() - x_all.min()) * 0.08
    xlim  = (x_all.min() - x_pad, x_all.max() + x_pad)

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5), sharex=True)

    for ri, sample in enumerate(SAMPLES):
        for ci, (param, col_label, (ylo, yhi)) in enumerate(PARAMS):
            ax = axes[ri, ci]

            # Scatter: all black; DSC filled circle, NMR/NMR2 open square
            for method in ['DSC', 'NMR', 'NMR2']:
                sub = df[(df['Method'] == method) & (df['Sample'] == sample)]
                if sub.empty:
                    continue
                x    = sub['inv_T'].to_numpy()
                y    = sub[param].to_numpy()
                yerr = sub[f'{param}_err'].to_numpy()
                marker = 'o' if method == 'DSC' else 's'
                mfc    = 'k' if method == 'DSC' else 'none'
                ax.errorbar(x, y, yerr=yerr, fmt=marker, color='k',
                            mfc=mfc, capsize=3, ms=5, lw=0.8, elinewidth=0.8)

            # Fit lines: weighted regression (same weights as compute_ea)
            for method in ['DSC', 'NMR']:
                sub = df_fit[(df_fit['Method'] == method) & (df_fit['Sample'] == sample)]
                if sub.empty or len(sub) < 2:
                    continue
                x    = sub['inv_T'].to_numpy()
                y    = sub[param].to_numpy()
                yerr = np.clip(sub[f'{param}_err'].to_numpy(), 1e-8, None)
                popt, _ = curve_fit(lambda X, s, b: s * X + b, x, y,
                                    sigma=yerr, absolute_sigma=True)
                x_fit = np.linspace(xlim[0], xlim[1], 150)
                ax.plot(x_fit, popt[0] * x_fit + popt[1],
                        color='k', ls=pu.METHOD_LS[method], lw=1.0, alpha=0.7)

            ax.set_xlim(xlim)
            ax.set_ylim(ylo, yhi)

            # Column header on top row only
            if ri == 0:
                ax.set_title(col_label, fontsize=10)
            # x-label on bottom row only
            if ri == 2:
                ax.set_xlabel(r'$1/T\,(\mathrm{K}^{-1})$')
            # y-label on left column only
            if ci == 0:
                ax.set_ylabel(r'$\ln\,k$')

    # Single legend in top-right panel
    handles = [
        Line2D([0], [0], marker='o', color='k', mfc='k',    ls='', ms=5, label='DSC'),
        Line2D([0], [0], marker='s', color='k', mfc='none', ls='', ms=5, label='NMR'),
    ]
    axes[0, 1].legend(handles=handles, loc='lower left', frameon=False, fontsize=8)

    fig.tight_layout()
    pu.savefig(fig, 'arrhenius')
    pu.write_provenance(
        __file__,
        datasets=['Posterior summaries — samples: EDA, DAP, DAB; methods: DSC, NMR, NMR2; all temperatures'],
        source_paths=[pu.DSC_CSV, pu.NMR_CSV],
    )
    plt.close(fig)

    _append_ea_analysis(df, df_fit)

def _append_ea_analysis(df, df_fit):
    """Compute Ea stats and append them to the provenance file."""
    DSC_NMR_TEMPS = {25, 33, 50}   # three lowest DSC temps, closest to NMR range

    lines = [
        '',
        'ACTIVATION ENERGY ANALYSIS',
        '  Values in kJ/mol.  % error = 100 × Ea_err / Ea.',
        '',
        '  Full fits (DSC: 6 temps; NMR: 3 temps):',
    ]
    for param, k_label in [('lnk1', 'k1'), ('lnk2', 'k2')]:
        lines.append(f'    {k_label}:')
        for method in ['DSC', 'NMR']:
            for sample in SAMPLES:
                try:
                    Ea, Ea_err = pu.compute_ea(df_fit, param, method, sample)
                    pct = 100.0 * abs(Ea_err / Ea)
                    lines.append(
                        f'      {method:3s} {sample}: Ea = {Ea:6.1f} ± {Ea_err:.1f} kJ/mol  ({pct:.1f}%)'
                    )
                except ValueError:
                    pass

    lines += [
        '',
        '  DSC restricted to 25, 33, 50 °C (closest DSC temps to NMR range):',
        '  [Isolates effect of temperature range vs temporal resolution on NMR uncertainty]',
    ]
    df_dsc_restricted = df[df['Temp_C'].isin(DSC_NMR_TEMPS)]
    for param, k_label in [('lnk1', 'k1'), ('lnk2', 'k2')]:
        lines.append(f'    {k_label}:')
        for sample in SAMPLES:
            try:
                Ea, Ea_err = pu.compute_ea(df_dsc_restricted, param, 'DSC', sample)
                pct = 100.0 * abs(Ea_err / Ea)
                lines.append(
                    f'      DSC {sample}: Ea = {Ea:6.1f} ± {Ea_err:.1f} kJ/mol  ({pct:.1f}%)'
                )
            except ValueError:
                pass

    prov_path = os.path.join(pu._figures_dir(), 'arrhenius.provenance.txt')
    with open(prov_path, 'a') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=None,
                        help='Redirect all output here instead of figures/ (does not touch the originals).')
    args = parser.parse_args()
    if args.outdir:
        pu.set_output_dir(args.outdir)
    main()
