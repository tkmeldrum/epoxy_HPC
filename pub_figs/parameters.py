"""4×2 parameter scatter: rows = ln k1 / ln k2 / m / n, cols = DSC / NMR."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pub_utils as pu

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

SAMPLE_MARKER = {'EDA': 'o', 'DAP': 's', 'DAB': 'D'}
SAMPLE_COLOR  = {'EDA': 'k', 'DAP': '#0072B2', 'DAB': '#D55E00'}
PARAMS = [
    ('lnk1', r'$\ln\,k_1$', (-25, 0)),
    ('lnk2', r'$\ln\,k_2$', (None, None)),
    ('m',    r'$m$',         (0, 3)),
    ('n',    r'$n$',         (0, 3)),
]
METHODS = [('DSC', ['DSC']), ('NMR', ['NMR', 'NMR2'])]  # NMR2 folds into NMR column
XLIM   = (20, 105)
JITTER = 1.5  # °C nudge for off-scale asterisks

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()

    fig, axes = plt.subplots(4, 2, figsize=(6.5, 7.5), sharex=True, sharey='row')

    for ri, (param, ylabel, (ylo, yhi)) in enumerate(PARAMS):
        for ci, (col_label, methods) in enumerate(METHODS):
            ax = axes[ri, ci]

            if ylo is not None:
                ax.set_ylim(ylo, yhi)

            jitter_idx = 0
            for sample in ['EDA', 'DAP', 'DAB']:
                color  = SAMPLE_COLOR[sample]
                marker = SAMPLE_MARKER[sample]
                for method in methods:
                    sub = df[(df['Method'] == method) & (df['Sample'] == sample)]
                    if sub.empty:
                        continue
                    x    = sub['Temp_C'].to_numpy()
                    y    = sub[param].to_numpy()
                    err_col = f'{param}_err'
                    yerr = sub[err_col].to_numpy() if err_col in sub.columns \
                        else np.zeros(len(sub))
                    mfc  = color  # all filled within their column
                    ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color,
                                mfc=mfc, capsize=3, ms=5, lw=0.8, elinewidth=0.8)

                    if yhi is not None:
                        for xi, (yi, ei) in enumerate(zip(y, yerr)):
                            if yi + ei > yhi or yi - ei < ylo:
                                nudge = JITTER * (1 if jitter_idx % 2 == 0 else -1)
                                ax.text(x[xi] + nudge, yhi, '*',
                                        ha='center', va='top', fontsize=9,
                                        color=color, fontweight='bold', clip_on=False)
                                jitter_idx += 1

            ax.set_xlim(XLIM)

            if ri == 0:
                ax.set_title(col_label)
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == 3:
                ax.set_xlabel('Temperature (°C)')

    # Legend in top-right panel: samples only (fill is redundant — all filled per column)
    legend_handles = [
        Line2D([0],[0], marker='o', color='k',       mfc='k',       ls='', ms=5, label='EDA'),
        Line2D([0],[0], marker='s', color='#0072B2', mfc='#0072B2', ls='', ms=5, label='DAP'),
        Line2D([0],[0], marker='D', color='#D55E00', mfc='#D55E00', ls='', ms=5, label='DAB'),
    ]
    axes[0, 1].legend(handles=legend_handles, frameon=False, fontsize=7,
                      loc='lower right')

    fig.tight_layout()
    pu.savefig(fig, 'parameters')
    pu.write_provenance(
        __file__,
        datasets=['Posterior summaries — samples: EDA, DAP, DAB; methods: DSC and NMR (NMR2 shown in NMR column); all temperatures; parameters: ln k1, ln k2, m, n'],
        source_paths=[pu.DSC_CSV, pu.NMR_CSV],
    )
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=None,
                        help='Redirect all output here instead of figures/ (does not touch the originals).')
    args = parser.parse_args()
    if args.outdir:
        pu.set_output_dir(args.outdir)
    main()
