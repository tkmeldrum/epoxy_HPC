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
from itertools import product as _iproduct
import pub_utils as pu

SAMPLE    = 'EDA'
TEMP_STR  = '33C'
TEMP_C    = 33
FIT_COLOR = '#D55E00'   # vermilion
FIT_LW    = 2.0
BAND_ALPHA = 0.25


def _r_val(row, default=1.0):
    # default=1.0: NMR posteriors have no r column (r was fixed at max(α_data)≈1.0,
    # not sampled). DSC rows always have r_median present so the default is unused.
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


def _param_corners(lo, hi):
    """Yield all 2^N parameter dicts at the corners of the lo/hi box."""
    keys = list(lo.keys())
    for choices in _iproduct(*[(lo[k], hi[k]) for k in keys]):
        yield dict(zip(keys, choices))


def _alpha_t_envelope(lo, hi, t_model):
    """Pointwise min/max of α(t) over all parameter-box corners."""
    curves = np.array([pu.solve_km(**p, t_eval_min=t_model)
                       for p in _param_corners(lo, hi)])
    return np.nanmin(curves, axis=0), np.nanmax(curves, axis=0)


def _dadt_envelope(lo, hi, n_pts=300):
    """Pointwise min/max of dα/dt vs α over all parameter-box corners.

    Uses a common α grid clipped below the minimum r across corners so the
    (r − α) term stays non-negative for every combination.
    """
    r_end   = min(lo['r'], hi['r'], 1.0) * 0.998
    a_curve = np.linspace(0, r_end, n_pts)
    curves  = []
    for p in _param_corners(lo, hi):
        k1, k2, m, n, r = p['k1'], p['k2'], p['m'], p['n'], p['r']
        epox = np.maximum(0.0, 1.0 - a_curve)
        rmia = np.maximum(0.0, r - a_curve)
        curves.append((k1 + k2 * a_curve**m) * epox**(n / 2) * rmia**(n / 2) * 60.0)
    curves = np.array(curves)
    return a_curve, np.min(curves, axis=0), np.max(curves, axis=0)


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

    t_max   = t_raw.max()
    t_start = t_raw.min()   # NMR: first scan > 0; DSC: ~0. Match fit's t0.
    lo, med, hi = get_param_bounds(df, method, sample, temp_c)

    fig, axes = plt.subplots(1, 2, figsize=pu.FIG_SIZE_1x2)
    fig.subplots_adjust(wspace=0.38)

    # ── α(t) — left ────────────────────────────────────────────────────────
    t_model          = np.linspace(t_start, t_max, 400)
    a_med            = pu.solve_km(**med, t_eval_min=t_model)
    a_band_lo, a_band_hi = _alpha_t_envelope(lo, hi, t_model)

    ax_a = axes[0]
    ax_a.plot(t_raw, a_raw, 'o', ms=3, color='k', mfc='k', lw=0,
              label='data', zorder=2)
    ax_a.fill_between(t_model, a_band_lo, a_band_hi,
                      color=FIT_COLOR, alpha=BAND_ALPHA, lw=0,
                      label='95% CI', zorder=3)
    ax_a.plot(t_model, a_med, color=FIT_COLOR, lw=FIT_LW,
              label='KM median', zorder=4)
    ax_a.set_xlim(0, t_max)
    ax_a.set_ylim(bottom=0)
    ax_a.set_ylabel(r'$\alpha$')
    ax_a.set_xlabel('time (min)')
    ax_a.legend(frameon=False, fontsize=7, loc='lower right')

    # ── dα/dt vs α — right ─────────────────────────────────────────────────
    a_curve, da_med          = dadt_vs_alpha(med)
    a_band, da_band_lo, da_band_hi = _dadt_envelope(lo, hi)

    ax_d = axes[1]
    ax_d.plot(a_raw, dadt_raw, 'o', ms=3, color='k', mfc='k', lw=0, zorder=2)
    ax_d.fill_between(a_band, da_band_lo, da_band_hi,
                      color=FIT_COLOR, alpha=BAND_ALPHA, lw=0, zorder=3)
    ax_d.plot(a_curve, da_med, color=FIT_COLOR, lw=FIT_LW, zorder=4)
    ax_d.set_xlim(0, min(1.0, a_raw.max()))
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

        t_start = t_raw.min()   # NMR row: match fit's t0; DSC row: ~0
        t_model          = np.linspace(t_start, t_max, 400)
        a_med            = pu.solve_km(**med, t_eval_min=t_model)
        a_band_lo, a_band_hi = _alpha_t_envelope(lo, hi, t_model)

        ax_a = axes[ri, 0]
        ax_a.plot(t_raw, a_raw, 'o', ms=3, color='k', mfc='k', lw=0,
                  label='data', zorder=2)
        ax_a.fill_between(t_model, a_band_lo, a_band_hi,
                          color=FIT_COLOR, alpha=BAND_ALPHA, lw=0,
                          label='95% CI', zorder=3)
        ax_a.plot(t_model, a_med, color=FIT_COLOR, lw=FIT_LW,
                  label='KM median', zorder=4)
        ax_a.set_xlim(xlim)
        ax_a.set_ylim(bottom=0)
        ax_a.set_xticks(x_ticks)
        ax_a.set_ylabel(r'$\alpha$')
        if ri == 1:
            ax_a.set_xlabel('time (min)')
        if ri == 0:
            ax_a.legend(frameon=False, fontsize=7, loc='lower right')

        a_curve, da_med          = dadt_vs_alpha(med)
        a_band, da_band_lo, da_band_hi = _dadt_envelope(lo, hi)

        ax_d = axes[ri, 1]
        ax_d.plot(a_raw, dadt_raw, 'o', ms=3, color='k', mfc='k', lw=0, zorder=2)
        ax_d.fill_between(a_band, da_band_lo, da_band_hi,
                          color=FIT_COLOR, alpha=BAND_ALPHA, lw=0, zorder=3)
        ax_d.plot(a_curve, da_med, color=FIT_COLOR, lw=FIT_LW, zorder=4)
        ax_d.set_xlim(0, min(1.0, a_raw.max()))
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
