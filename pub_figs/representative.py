"""2×2 representative cure figure.

Rows: top = DSC EDA 33 °C, bottom = NMR EDA 33 °C
Cols: left  = α(t)       with 95% CI band
      right = dα/dt vs α with 95% CI band

Posterior band uses lo-lo / hi-hi parameter bounds from the posterior
summary CSV — an approximation of the 95% credible envelope
(full MCMC chains are not persisted for these runs).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import pub_utils as pu

SAMPLE    = 'EDA'
TEMP_STR  = '33C'
TEMP_C    = 33
FIT_COLOR = '#D55E00'   # vermilion
FIT_LW    = 2.0
BAND_ALPHA = 0.25


def _r_val(row, default=2.0):
    v = row.get('r', np.nan)
    return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v


def get_param_bounds(df, method, sample, temp_c):
    """Return (lo, med, hi) param dicts from posterior CI bounds.

    lo-lo and hi-hi are not true credible envelopes (that requires chains),
    but give a conservative bound on the 95% CI range.
    """
    row  = df[(df['Method'] == method) & (df['Sample'] == sample) &
              (df['Temp_C'] == temp_c)].iloc[0]
    r_m  = _r_val(row)
    r_lo = row.get('r_CI_lower', r_m); r_lo = r_m if (isinstance(r_lo, float) and np.isnan(r_lo)) else r_lo
    r_hi = row.get('r_CI_upper', r_m); r_hi = r_m if (isinstance(r_hi, float) and np.isnan(r_hi)) else r_hi

    med = dict(k1=10**row['log_k1_median'],   k2=10**row['log_k2_median'],
               m=row['m_median'],              n=row['n_median'],   r=r_m)
    lo  = dict(k1=10**row['log_k1_CI_lower'], k2=10**row['log_k2_CI_lower'],
               m=row['m_CI_lower'],            n=row['n_CI_lower'], r=r_lo)
    hi  = dict(k1=10**row['log_k1_CI_upper'], k2=10**row['log_k2_CI_upper'],
               m=row['m_CI_upper'],            n=row['n_CI_upper'], r=r_hi)
    return lo, med, hi


def dadt_vs_alpha(params, n_pts=300):
    """Analytical KM dα/dt as a function of α (min⁻¹)."""
    k1, k2, m, n, r = params['k1'], params['k2'], params['m'], params['n'], params['r']
    a_end = min(r, 1.0) * 0.998
    av    = np.linspace(0, a_end, n_pts)
    epox  = np.maximum(0.0, 1.0 - av)
    return av, (k1 + k2 * av**m) * epox**(n / 2) * (r - av)**(n / 2) * 60.0


def make_rep_figure(df, sample, temp_str, temp_c, method, stem=None):
    """1×2 figure: α(t) with CI band (left), dα/dt vs α (right).

    method: 'DSC', 'NMR', or 'NMR2'
    stem: path relative to pub_utils._FIGURES (without extension).
          Defaults to 'SI_figures/rep_{method}_{sample}_{temp_str}'.
    """
    if stem is None:
        stem = f'SI_figures/rep_{method}_{sample}_{temp_str}'

    if method == 'DSC':
        t_raw, a_raw, dadt_raw = pu.load_dsc_raw(sample, temp_str)
        step = max(1, len(t_raw) // 300)
        t_raw, a_raw, dadt_raw = t_raw[::step], a_raw[::step], dadt_raw[::step]
    else:
        # NMR2 raw data is stored under 'DAP2' in all_samples.csv
        raw_sample = 'DAP2' if method == 'NMR2' else sample
        t_raw, a_raw = pu.load_nmr_raw(raw_sample, temp_str)
        dadt_raw = np.gradient(a_raw, t_raw)

    t_max  = t_raw.max()
    lo, med, hi = get_param_bounds(df, method, sample, temp_c)

    fig, axes = plt.subplots(1, 2, figsize=pu.FIG_SIZE_1x2)
    fig.subplots_adjust(wspace=0.38)

    # ── α(t) — left ────────────────────────────────────────────────────────
    t_model = np.linspace(0, t_max, 400)
    a_med   = pu.solve_km(**med, t_eval_min=t_model)
    a_lo    = pu.solve_km(**lo,  t_eval_min=t_model)
    a_hi    = pu.solve_km(**hi,  t_eval_min=t_model)

    ax_a = axes[0]
    ax_a.plot(t_raw, a_raw, 'o', ms=3, color='k', mfc='k', lw=0,
              label='data', zorder=4)
    ax_a.fill_between(t_model, a_lo, a_hi,
                      color=FIT_COLOR, alpha=BAND_ALPHA, lw=0,
                      label='95% CI', zorder=2)
    ax_a.plot(t_model, a_med, color=FIT_COLOR, lw=FIT_LW,
              label='KM median', zorder=3)
    ax_a.set_xlim(0, t_max)
    ax_a.set_ylabel(r'$\alpha$')
    ax_a.set_xlabel('time (min)')
    ax_a.legend(frameon=False, fontsize=7, loc='upper left')

    # ── dα/dt vs α — right ─────────────────────────────────────────────────
    a_curve, da_med = dadt_vs_alpha(med)
    _,       da_lo  = dadt_vs_alpha(lo)
    _,       da_hi  = dadt_vs_alpha(hi)

    ax_d = axes[1]
    ax_d.plot(a_raw, dadt_raw, 'o', ms=3, color='k', mfc='k', lw=0, zorder=4)
    ax_d.fill_between(a_curve, da_lo, da_hi,
                      color=FIT_COLOR, alpha=BAND_ALPHA, lw=0, zorder=2)
    ax_d.plot(a_curve, da_med, color=FIT_COLOR, lw=FIT_LW, zorder=3)
    ax_d.set_ylabel(r'd$\alpha$/d$t$ (min$^{-1}$)')
    ax_d.set_xlabel(r'$\alpha$')

    pu.savefig(fig, stem)
    plt.close(fig)
    return stem


def main():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    })
    pu.snapshot_data()
    df = pu.load_posteriors()

    # ── Load raw data ──────────────────────────────────────────────────────
    t_dsc, a_dsc, da_dsc = pu.load_dsc_raw(SAMPLE, TEMP_STR)
    step = max(1, len(t_dsc) // 300)
    t_dsc, a_dsc, da_dsc = t_dsc[::step], a_dsc[::step], da_dsc[::step]

    t_nmr, a_nmr = pu.load_nmr_raw(SAMPLE, TEMP_STR)
    da_nmr = np.gradient(a_nmr, t_nmr)

    t_max   = min(t_dsc.max(), t_nmr.max())
    xlim    = (0, t_max)
    x_ticks = range(0, 301, 50)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
    fig.subplots_adjust(hspace=0.38, wspace=0.38)

    rows = [
        ('DSC', t_dsc, a_dsc, da_dsc),
        ('NMR', t_nmr, a_nmr, da_nmr),
    ]

    for ri, (method, t_raw, a_raw, dadt_raw) in enumerate(rows):
        lo, med, hi = get_param_bounds(df, method, SAMPLE, TEMP_C)

        t_model = np.linspace(0, t_max, 400)
        a_med   = pu.solve_km(**med, t_eval_min=t_model)
        a_lo    = pu.solve_km(**lo,  t_eval_min=t_model)
        a_hi    = pu.solve_km(**hi,  t_eval_min=t_model)

        ax_a = axes[ri, 0]
        ax_a.plot(t_raw, a_raw, 'o', ms=3, color='k', mfc='k', lw=0,
                  label='data', zorder=4)
        ax_a.fill_between(t_model, a_lo, a_hi,
                          color=FIT_COLOR, alpha=BAND_ALPHA, lw=0,
                          label='95% CI', zorder=2)
        ax_a.plot(t_model, a_med, color=FIT_COLOR, lw=FIT_LW,
                  label='KM median', zorder=3)
        ax_a.set_xlim(xlim)
        ax_a.set_xticks(x_ticks)
        ax_a.set_ylabel(r'$\alpha$')
        if ri == 1:
            ax_a.set_xlabel('time (min)')
        if ri == 0:
            ax_a.legend(frameon=False, fontsize=7, loc='upper left')

        a_curve, da_med = dadt_vs_alpha(med)
        _,       da_lo  = dadt_vs_alpha(lo)
        _,       da_hi  = dadt_vs_alpha(hi)

        ax_d = axes[ri, 1]
        ax_d.plot(a_raw, dadt_raw, 'o', ms=3, color='k', mfc='k', lw=0, zorder=4)
        ax_d.fill_between(a_curve, da_lo, da_hi,
                          color=FIT_COLOR, alpha=BAND_ALPHA, lw=0, zorder=2)
        ax_d.plot(a_curve, da_med, color=FIT_COLOR, lw=FIT_LW, zorder=3)
        ax_d.set_ylabel(r'd$\alpha$/d$t$ (min$^{-1}$)')
        ax_d.set_xlabel(r'$\alpha$')

    pu.savefig(fig, 'representative')
    pu.write_provenance(
        __file__,
        datasets=[
            f'Posterior summaries — sample: {SAMPLE}, temp: {TEMP_STR}; methods: DSC and NMR',
            f'DSC raw time-series — sample: {SAMPLE}, temp: {TEMP_STR} (alpha, dα/dt)',
            f'NMR raw time-series — sample: {SAMPLE}, temp: {TEMP_STR} (alpha vs elapsed time)',
        ],
        source_paths=[pu.DSC_CSV, pu.NMR_CSV, pu.MAT_FILE, pu.NMR_RAW],
    )
    plt.close(fig)


if __name__ == '__main__':
    main()
