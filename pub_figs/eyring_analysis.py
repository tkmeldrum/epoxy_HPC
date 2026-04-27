"""Eyring / transition-state theory analysis of epoxy-amine cure kinetics.

Background and decisions
------------------------
The Kamal-Malkin (KM) model fitted to DSC and NMR isothermal-cure data yields
two rate constants k1 and k2 (in min⁻¹) at each temperature T.  Arrhenius
analysis of those constants lives in arrhenius.py.  This script performs the
complementary Eyring / activated-complex (TST) analysis.

Eyring equation (transmission coefficient κ = 1, no tunnelling correction):

    k = (k_B T / h) · exp(−ΔG‡ / RT)
      = (k_B T / h) · exp(ΔS‡ / R) · exp(−ΔH‡ / RT)

Rearranging for linear regression:

    ln(k_s / T) = −(ΔH‡/R) · (1/T)  +  ΔS‡/R + ln(k_B/h)
                    ↑ slope                  ↑ intercept

where k_s = k_min / 60 (rate constant in s⁻¹, making the intercept
dimensionally consistent with ln(k_B/h) whose units are ln(s⁻¹ K⁻¹)).

Physical constants (NIST CODATA 2018):
  k_B  = 1.380649 × 10⁻²³ J K⁻¹
  h    = 6.62607015 × 10⁻³⁴ J s
  R    = 8.3145 J mol⁻¹ K⁻¹
  ln(k_B/h) ≈ 23.760  [ln(s⁻¹ K⁻¹)]

ΔH‡ vs Ea
----------
For condensed-phase reactions ΔH‡ ≈ ΔU‡ (negligible PV work, no change in
moles of gas).  However, the relationship between Ea (Arrhenius) and ΔH‡
(Eyring) is NOT Ea = ΔH‡.  Differentiating ln(k) from the Arrhenius form and
ln(k/T) from the Eyring form with respect to 1/T reveals that the offset
arises purely from the factor of T in the k_B T/h pre-exponential:

    d ln k / d(1/T) = −Ea / R               [Arrhenius]
    d ln(k/T) / d(1/T) = −ΔH‡ / R          [Eyring, if ΔH‡ indep. of T]
    → d ln k / d(1/T) = −ΔH‡/R − d ln T / d(1/T) = −ΔH‡/R − (−T)

    ∴  Ea = ΔH‡ + RT    (≈ 2.5 kJ/mol at 25 °C, ≈ 2.7 kJ/mol at 50 °C)

This holds for both gas and condensed phases; it is not a PV correction.
The sanity-check column (Ea − ΔH‡) should equal RT at the mean temperature.

Uncertainty propagation
-----------------------
Regression uncertainties come from scipy.optimize.curve_fit with
absolute_sigma=True; the sigma values are the posterior CI half-widths on lnk.
For ΔG‡ = ΔH‡ − Tref · ΔS‡ the full off-diagonal covariance is used:

    σ²(ΔG‡) = (R/1000)² · [pcov₀₀  +  Tref² · pcov₁₁  +  2·Tref · pcov₀₁]

NMR2 note
---------
The NMR posterior CSV contains two DAP datasets (DAP and DAP2/NMR2).
Following arrhenius.py convention, NMR2 is merged into NMR before fitting so
that DAP/NMR Eyring fits use all available temperature points.

Outputs (all separate from arrhenius.py outputs):
  figures/eyring.pdf / .png         — 3×2 Eyring plot
  figures/eyring.provenance.txt     — data hashes and timestamp
  <Epoxy-Kinetics-2025>/Ea_table.tex — LaTeX table: Ea, ΔH‡, ΔS‡ for DSC & NMR
"""
import sys, os
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

# ── Physical constants (NIST CODATA 2018) ──────────────────────────────────────
KB       = 1.380649e-23    # J / K
H_PLANCK = 6.62607015e-34  # J · s
R        = 8.3145          # J / (mol · K)
LN_KB_H  = np.log(KB / H_PLANCK)  # ≈ 23.760  [ln(s⁻¹ K⁻¹)]
LN60     = np.log(60.0)            # unit factor: k [min⁻¹] → k [s⁻¹]
TREF     = 298.15                  # K (25 °C reference for ΔG‡)

SAMPLES = ['EDA', 'DAP', 'DAB']
PARAMS  = [('lnk1', r'$k_1$'), ('lnk2', r'$k_2$')]

_HERE           = os.path.dirname(os.path.abspath(__file__))
LATEX_TABLE_OUT = os.path.normpath(
    os.path.join(_HERE, '..', '..', 'Epoxy-Kinetics-2025', 'Ea_table.tex')
)


# ── Core Eyring regression ─────────────────────────────────────────────────────
def compute_eyring(df, param, method, sample):
    """Weighted Eyring regression for one (method, sample, param) group.

    param   : 'lnk1' or 'lnk2'
    Returns a dict of thermodynamic parameters and fit diagnostics,
    or None if fewer than 2 data points are available.
    """
    mask = (df['Method'] == method) & (df['Sample'] == sample)
    sub  = df[mask].dropna(subset=[param, f'{param}_err']).copy()
    if len(sub) < 2:
        return None

    T_K  = sub['Temp_C'].to_numpy() + 273.15
    x    = 1.0 / T_K
    y    = sub[param].to_numpy() - LN60 - np.log(T_K)   # ln(k_s / T)
    yerr = np.clip(sub[f'{param}_err'].to_numpy(), 1e-8, None)

    popt, pcov = curve_fit(
        lambda X, slope, intercept: slope * X + intercept,
        x, y, sigma=yerr, absolute_sigma=True,
    )
    slope, intercept = popt

    dH = -slope * R / 1000.0          # kJ / mol
    dS = (intercept - LN_KB_H) * R    # J / (mol · K)
    dG = dH - TREF * dS / 1000.0      # kJ / mol

    # Full-covariance propagation:  ∂ΔG/∂slope = −R/1000,  ∂ΔG/∂intercept = −Tref·R/1000
    J      = np.array([-R / 1000.0, -TREF * R / 1000.0])
    dH_err = np.sqrt(pcov[0, 0]) * R / 1000.0
    dS_err = np.sqrt(pcov[1, 1]) * R
    dG_err = np.sqrt(J @ pcov @ J)

    return dict(
        dH=dH, dH_err=dH_err,
        dS=dS, dS_err=dS_err,
        dG=dG, dG_err=dG_err,
        slope=slope, intercept=intercept, pcov=pcov,
        x=x, y=y, yerr=yerr, n_pts=len(sub),
    )


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _fmt(val, err, decimals=1):
    """Produce a siunitx \\num{val \\pm err} string."""
    fmt = f'.{decimals}f'
    return rf'\num{{{val:{fmt}} \pm {err:{fmt}}}}'


# ── Eyring plot ────────────────────────────────────────────────────────────────
def make_eyring_plot(df, results_eyring):
    """3×2 grid of Eyring plots.

    Rows: EDA / DAP / DAB.  Cols: k1 / k2.
    Scatter: DSC filled circles, NMR/NMR2 open squares — black with error bars.
    Regression lines: DSC dashed, NMR dotted — black.
    """
    x_all = df['inv_T'].to_numpy()
    x_pad = (x_all.max() - x_all.min()) * 0.08
    xlim  = (x_all.min() - x_pad, x_all.max() + x_pad)
    x_fit = np.linspace(xlim[0], xlim[1], 150)

    fig, axes = plt.subplots(3, 2, figsize=pu.FIG_SIZE_2x2, sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.38)

    for ri, sample in enumerate(SAMPLES):
        for ci, (param, col_label) in enumerate(PARAMS):
            ax = axes[ri, ci]

            # Scatter — raw (df includes NMR2 as separate method)
            ax_y_lo, ax_y_hi = [], []
            for method in ['DSC', 'NMR', 'NMR2']:
                sub = df[(df['Method'] == method) & (df['Sample'] == sample)]
                if sub.empty:
                    continue
                T_K  = sub['Temp_C'].to_numpy() + 273.15
                x    = 1.0 / T_K
                y    = sub[param].to_numpy() - LN60 - np.log(T_K)
                yerr = np.clip(sub[f'{param}_err'].to_numpy(), 1e-8, None)
                ax_y_lo.extend(y - yerr)
                ax_y_hi.extend(y + yerr)
                marker = 'o' if method == 'DSC' else 's'
                mfc    = 'k' if method == 'DSC' else 'none'
                ax.errorbar(x, y, yerr=yerr, fmt=marker, color='k',
                            mfc=mfc, capsize=3, ms=5, lw=0.8, elinewidth=0.8)

            # Regression lines (merged NMR2 → NMR, stored in results)
            for method in ['DSC', 'NMR']:
                r = results_eyring.get(method, {}).get(sample, {}).get(param)
                if r is None:
                    continue
                ax.plot(x_fit, r['slope'] * x_fit + r['intercept'],
                        color='k', ls=pu.METHOD_LS[method], lw=1.0, alpha=0.7)

            ax.set_xlim(xlim)
            # Robust y-limits: exclude the single most extreme lower outlier so
            # one huge NMR error bar doesn't compress the rest of the panel.
            # y-max is anchored at 0 (ln(k_s/T) is always negative for these k).
            if ax_y_lo:
                lo_sorted = sorted(ax_y_lo)
                y_ref_lo  = lo_sorted[1] if len(lo_sorted) > 1 else lo_sorted[0]
                pad = 0.08 * abs(max(ax_y_hi) - y_ref_lo)
                ax.set_ylim(y_ref_lo - pad, 0)
            if ri == 0:
                ax.set_title(col_label, fontsize=10)
            if ri == 2:
                ax.set_xlabel(r'$1/T\,(\mathrm{K}^{-1})$')
            if ci == 0:
                ax.set_ylabel(r'$\ln(k_{\mathrm{s}}/T)$')

    handles = [
        Line2D([0], [0], marker='o', color='k', mfc='k',    ls='', ms=5, label='DSC'),
        Line2D([0], [0], marker='s', color='k', mfc='none', ls='', ms=5, label='NMR'),
    ]
    axes[0, 1].legend(handles=handles, loc='lower left', frameon=False, fontsize=8)
    return fig


# ── Console summary ────────────────────────────────────────────────────────────
def print_summary(results_ea, results_eyring):
    """Print Ea / ΔH‡ / ΔS‡ / ΔG‡ table and Ea−ΔH‡ vs RT sanity check."""
    cols = ('Method', 'Sample', 'Param',
            'Ea (kJ/mol)', 'ΔH‡ (kJ/mol)', 'ΔS‡ (J/mol/K)',
            'ΔG‡ (kJ/mol)', 'Ea−ΔH‡', 'RT', 'n')
    widths = (7, 7, 6, 18, 18, 18, 18, 10, 8, 3)
    header = '  '.join(f'{c:<{w}}' for c, w in zip(cols, widths))
    print(header)
    print('─' * len(header))

    for method in ['DSC', 'NMR']:
        for sample in SAMPLES:
            for param, _ in PARAMS:
                ea_r  = results_ea.get(method, {}).get(sample, {}).get(param)
                ey_r  = results_eyring.get(method, {}).get(sample, {}).get(param)

                ea_str = (f'{ea_r[0]:6.1f} ± {ea_r[1]:5.1f}' if ea_r
                          else '            ---')
                if ey_r:
                    T_mean = np.mean(1.0 / ey_r['x'])
                    RT     = R * T_mean / 1000.0
                    diff   = (ea_r[0] - ey_r['dH']) if ea_r else float('nan')
                    dH_str = f"{ey_r['dH']:6.1f} ± {ey_r['dH_err']:5.1f}"
                    dS_str = f"{ey_r['dS']:6.1f} ± {ey_r['dS_err']:5.1f}"
                    dG_str = f"{ey_r['dG']:6.1f} ± {ey_r['dG_err']:5.1f}"
                    diff_s = f'{diff:6.2f}' if np.isfinite(diff) else '   ---'
                    RT_s   = f'{RT:6.2f}'
                    n_s    = str(ey_r['n_pts'])
                else:
                    dH_str = dS_str = dG_str = diff_s = RT_s = n_s = '---'

                vals = (method, sample, param,
                        ea_str, dH_str, dS_str, dG_str, diff_s, RT_s, n_s)
                print('  '.join(f'{v:<{w}}' for v, w in zip(vals, widths)))
        print()


# ── LaTeX table ────────────────────────────────────────────────────────────────
def write_latex_table(results_ea, results_eyring):
    """Overwrite Ea_table.tex with Ea, ΔH‡, and ΔS‡ for DSC and NMR."""
    def ea_cell(r):
        return _fmt(r[0], r[1]) if r else r'---'

    def ey_cell(r, field, err_field, dec=1):
        return _fmt(r[field], r[err_field], dec) if r else r'---'

    param_labels = {'lnk1': r'$k_1$', 'lnk2': r'$k_2$'}

    lines = [
        r'\begin{table}[!ht]',
        (r'\caption{Arrhenius activation energies ($E_a$) and Eyring activation'
         r' parameters ($\Delta H^\ddagger$, $\Delta S^\ddagger$) from'
         r' Kamal-Malkin rate constants $k_1$ and $k_2$ for DGEBA cured with'
         r' ethylenediamine (EDA), 1,3-diaminopropane (DAP), and'
         r' 1,4-diaminobutane (DAB).'
         r' Energies in kJ\,mol$^{-1}$;'
         r' $\Delta S^\ddagger$ in J\,mol$^{-1}$\,K$^{-1}$.'
         r' Reference temperature $T_\mathrm{ref} = 298.15$\,K for $\Delta G^\ddagger$.'
         r' Values are median $\pm$ half-width of the 95\,\% credible interval.'
         r' DSC columns use \qtylist[list-units=single]{25;33;50;60;80;100}{\celsius};'
         r' NMR columns use only \qtylist[list-units=single]{25;33;40}{\celsius}'
         r' and carry larger uncertainties.}'
         ),
        r'\label{tab:Ea_eyring}',
        r'    \centering',
        r'    \begin{tabular}{cc|cc|cc|cc}',
        (r'        \multirow{2}{*}{Sample} & \multirow{2}{*}{Rate const.}'
         r' & \multicolumn{2}{c|}{$E_a$}'
         r' & \multicolumn{2}{c|}{$\Delta H^\ddagger$}'
         r' & \multicolumn{2}{c}{$\Delta S^\ddagger$} \\'),
        r'        & & DSC & NMR & DSC & NMR & DSC & NMR \\ \midrule',
    ]

    for si, sample in enumerate(SAMPLES):
        last_sample = (si == len(SAMPLES) - 1)
        for pi, (param, _) in enumerate(PARAMS):
            last_row = last_sample and (pi == len(PARAMS) - 1)

            ea_dsc = results_ea.get('DSC', {}).get(sample, {}).get(param)
            ea_nmr = results_ea.get('NMR', {}).get(sample, {}).get(param)
            ey_dsc = results_eyring.get('DSC', {}).get(sample, {}).get(param)
            ey_nmr = results_eyring.get('NMR', {}).get(sample, {}).get(param)

            row_label = (rf'        \multirow{{2}}{{*}}{{{sample}}}'
                         if pi == 0 else r'        ~')
            cells = [
                row_label,
                param_labels[param],
                ea_cell(ea_dsc),
                ea_cell(ea_nmr),
                ey_cell(ey_dsc, 'dH', 'dH_err'),
                ey_cell(ey_nmr, 'dH', 'dH_err'),
                ey_cell(ey_dsc, 'dS', 'dS_err'),
                ey_cell(ey_nmr, 'dS', 'dS_err'),
            ]
            terminator = r' \\' if last_row else (
                r' \\' if pi == 0 else r' \\ \midrule'
            )
            lines.append(' & '.join(cells) + terminator)

    lines += [
        r'    \end{tabular}',
        r'\end{table}',
    ]

    dest_dir = os.path.dirname(LATEX_TABLE_OUT)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(LATEX_TABLE_OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'LaTeX table: {LATEX_TABLE_OUT}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    })
    pu.snapshot_data()
    df = pu.load_posteriors()

    # Merge NMR2 → NMR so DAP benefits from both NMR datasets (mirrors arrhenius.py)
    df_fit = df.copy()
    df_fit.loc[df_fit['Method'] == 'NMR2', 'Method'] = 'NMR'

    # ── Regressions ───────────────────────────────────────────────────────────
    results_ea     = {}
    results_eyring = {}

    for method in ['DSC', 'NMR']:
        results_ea[method]     = {}
        results_eyring[method] = {}
        for sample in SAMPLES:
            results_ea[method][sample]     = {}
            results_eyring[method][sample] = {}
            for param, _ in PARAMS:
                try:
                    ea, ea_err = pu.compute_ea(df_fit, param, method, sample)
                    results_ea[method][sample][param] = (ea, ea_err)
                except ValueError:
                    results_ea[method][sample][param] = None

                results_eyring[method][sample][param] = compute_eyring(
                    df_fit, param, method, sample
                )

    # ── Plot (original df so NMR2 scatter points appear separately) ───────────
    fig = make_eyring_plot(df, results_eyring)
    pu.savefig(fig, 'eyring')
    plt.close(fig)

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(results_ea, results_eyring)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    write_latex_table(results_ea, results_eyring)

    pu.write_provenance(
        __file__,
        datasets=[
            'Posterior summaries — all samples (EDA, DAP, DAB); '
            'methods DSC and NMR; all temperatures',
        ],
        source_paths=[pu.DSC_CSV, pu.NMR_CSV],
    )


if __name__ == '__main__':
    main()
