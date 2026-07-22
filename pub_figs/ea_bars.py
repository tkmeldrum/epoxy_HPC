"""Grouped bar chart: Ea for k1 (left) and k2 (right)."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import pub_utils as pu

# All sans-serif for this figure
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

SAMPLES = ['EDA', 'DAP', 'DAB']
METHODS = ['DSC', 'NMR']
METHOD_BAR_COLOR = {'DSC': '#0072B2', 'NMR': '#D55E00'}

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()
    df_fit = df.copy()
    df_fit.loc[df_fit['Method'] == 'NMR2', 'Method'] = 'NMR'

    # Pre-collect all Ea values
    ea = {}  # (param, method, sample) -> (Ea, err)
    for param in ['lnk1', 'lnk2']:
        for method in METHODS:
            for sample in SAMPLES:
                try:
                    ea[(param, method, sample)] = pu.compute_ea(df_fit, param, method, sample)
                except ValueError:
                    ea[(param, method, sample)] = (np.nan, np.nan)

    # ylim from k2 values (excluding NaN)
    k2_pos = [ea[('lnk2', m, s)][0] for m in METHODS for s in SAMPLES
              if not np.isnan(ea[('lnk2', m, s)][0])]
    y_top = max(k2_pos) * 1.18
    ylim  = (0, y_top)

    fig, axes = plt.subplots(1, 2, figsize=pu.FIG_SIZE_BARS)
    x     = np.arange(len(SAMPLES))
    width = 0.35

    for j, (param, title) in enumerate([('lnk1', r'$k_1$'), ('lnk2', r'$k_2$')]):
        ax = axes[j]
        for i, method in enumerate(METHODS):
            Eas, errs = [], []
            for sample in SAMPLES:
                Ea, err = ea[(param, method, sample)]
                # Negative or NaN → zero-height bar; annotated separately
                Eas.append(max(Ea, 0.0) if not np.isnan(Ea) else 0.0)
                # Suppress error bar when bar is clipped (Ea ≤ 0 or NaN)
                errs.append(err if (not np.isnan(Ea) and Ea > 0) else 0.0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, Eas, width, yerr=errs,
                   label=method, color=METHOD_BAR_COLOR[method],
                   capsize=4, error_kw=dict(lw=1.0), zorder=2)

        # k1-specific out-of-range annotations
        if param == 'lnk1':
            for i, method in enumerate(METHODS):
                offset = (i - 0.5) * width
                for si, sample in enumerate(SAMPLES):
                    Ea, err = ea[(param, method, sample)]
                    # EDA NMR k1 (err ≈ 354) and DAP NMR k1 (err ≈ 333): CI off scale
                    if method == 'NMR' and sample in ('EDA', 'DAP'):
                        # Place at top edge, nudged right so it clears the error bar line
                        ax.text(x[si] + offset + width * 0.35, y_top * 0.97, '*',
                                ha='center', va='top', fontsize=11,
                                color=METHOD_BAR_COLOR[method], fontweight='bold',
                                clip_on=False)
                    # DAB NMR k1 ≈ −119 kJ/mol: below zero, show downward arrow
                    if method == 'NMR' and sample == 'DAB' and (np.isnan(Ea) or Ea < 0):
                        ax.annotate(
                            '', xy=(x[si] + offset, -y_top * 0.06),
                            xytext=(x[si] + offset, 0),
                            annotation_clip=False,
                            arrowprops=dict(arrowstyle='->', lw=1.2,
                                            color=METHOD_BAR_COLOR[method]),
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(SAMPLES)
        ax.set_title(title)
        ax.set_ylim(ylim)
        # ylabel on left only
        if j == 0:
            ax.set_ylabel(r'$E_\mathrm{a}$ (kJ mol$^{-1}$)')
        # legend on right only
        if j == 1:
            ax.legend(frameon=False)

    fig.tight_layout()
    pu.savefig(fig, 'ea_bars')
    pu.write_provenance(
        __file__,
        datasets=['Posterior summaries — samples: EDA, DAP, DAB; methods: DSC, NMR (NMR2 folded into NMR); all temperatures; Ea derived by weighted Arrhenius regression'],
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
