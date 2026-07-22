"""Generate LaTeX KM parameter table for SI.

Matches structure of Epoxy-Kinetics-2025/SI/KM_with_r_table.tex.
NMR r = max(α_data) fixed per dataset; DSC r from posterior median.
DAP2 shown as NMR2 sub-block under DAP.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pub_utils as pu

SAMPLE_ORDER = ['EDA', 'DAP', 'DAB']
DSC_TEMPS    = [25, 33, 50, 60, 80, 100]
NMR_TEMPS    = [25, 33, 40, 60]

def fmt_val(median, ci_lo, ci_hi, scale=1.0):
    """Format median ± halfCI in linear space, scaled."""
    val     = median * scale
    half_ci = abs(ci_hi - ci_lo) / 2.0 * scale
    if half_ci == 0:
        return rf'\num{{{val:.3g}}}'
    decimals = max(0, -int(math.floor(math.log10(half_ci))) + 1)
    fmt      = f'{{:.{decimals}f}}'
    return rf'\num{{{fmt.format(val)} \pm {fmt.format(half_ci)}}}'

def fmt_k(log_median, log_lo, log_hi, scale_exp):
    """k values: convert from log10 to linear, scale to 10^scale_exp s^-1."""
    scale = 10 ** (-scale_exp)  # e.g. scale_exp=-6 → scale=1e6
    k_med = 10 ** log_median * scale
    k_lo  = 10 ** log_lo * scale
    k_hi  = 10 ** log_hi * scale
    half  = abs(k_hi - k_lo) / 2.0
    if half == 0:
        return rf'\num{{{k_med:.3g}}}'
    decimals = max(0, -int(math.floor(math.log10(half))) + 1)
    fmt      = f'{{:.{decimals}f}}'
    return rf'\num{{{fmt.format(k_med)} \pm {fmt.format(half)}}}'

def get_row(df, method, sample, temp_c):
    sub = df[(df['Method'] == method) & (df['Sample'] == sample) &
             (df['Temp_C'] == temp_c)]
    if sub.empty:
        return None
    return sub.iloc[0]

def main():
    df = pu.load_posteriors()

    lines = [
        r'\begin{table}[!ht]',
        r'\caption{Modified Kamal-Malkin fits of DGEBA with '
        r'ethylenediamine (EDA), 1,3-diaminopropane (DAP), and '
        r'1,4-diaminobutane (DAB). Reported values are posterior medians; '
        r'uncertainties represent the 95\% credible intervals from the '
        r'Bayesian analysis. For DSC, $r$ is a free parameter per temperature '
        r'(equal to $\alpha_\infty$). For NMR, $r$ is fixed to the observed '
        r'maximum conversion per dataset ($r = \max(\alpha_\mathrm{data}) '
        r'=1.0$) rather than the stoichiometric ratio.}',
        r'\label{tab:KM_with_r}',
        r'    \centering',
        r'    \begin{tabular}{c|cc ccccc}',
        r'        Sample & Method & Temp ($^\circ$C) & '
        r'$k_1/10^{-6}$ [s$^{-1}$] & $k_2/10^{-3}$ [s$^{-1}$] & '
        r'$m$ & $n$ & $r$ \\ \midrule',
    ]

    for si, sample in enumerate(SAMPLE_ORDER):
        # NMR2/DAP2 was never measured at every NMR_TEMPS entry (e.g. no 60C run) --
        # count what's actually present rather than assuming it mirrors NMR_TEMPS.
        nmr2_temps_present = (
            [t for t in NMR_TEMPS if get_row(df, 'NMR2', 'DAP', t) is not None]
            if sample == 'DAP' else []
        )
        n_total = 6 + len(NMR_TEMPS) + len(nmr2_temps_present)

        # DSC rows
        for ti, temp in enumerate(DSC_TEMPS):
            row = get_row(df, 'DSC', sample, temp)
            if row is None:
                continue
            k1 = fmt_k(row['log_k1_median'], row['log_k1_CI_lower'],
                       row['log_k1_CI_upper'], -6)
            k2 = fmt_k(row['log_k2_median'], row['log_k2_CI_lower'],
                       row['log_k2_CI_upper'], -3)
            m  = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
            n  = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
            r  = fmt_val(row['r_median'], row['r_CI_lower'], row['r_CI_upper'])

            if ti == 0:
                prefix = (rf'\multirow{{{n_total}}}{{*}}{{{sample}}} & '
                          rf'\multirow{{6}}{{*}}{{DSC}} & ')
            else:
                prefix = r'~ & ~ & '
            lines.append(f'        {prefix}{temp} & {k1} & {k2} & {m} & {n} & {r} \\\\')

        lines.append(r'        \cline{2-8}')

        if sample == 'DAP':
            # Group by temperature: NMR row, plus NMR2 row only where it exists.
            nmr_n = len(NMR_TEMPS) + len(nmr2_temps_present)
            is_first_row = True
            for ti, temp in enumerate(NMR_TEMPS):
                methods_here = ['NMR'] + (['NMR2'] if temp in nmr2_temps_present else [])
                for ri, method in enumerate(methods_here):
                    row = get_row(df, method, sample, temp)
                    if row is None:
                        continue
                    k1 = fmt_k(row['log_k1_median'], row['log_k1_CI_lower'],
                               row['log_k1_CI_upper'], -6)
                    k2 = fmt_k(row['log_k2_median'], row['log_k2_CI_lower'],
                               row['log_k2_CI_upper'], -3)
                    m  = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
                    n  = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
                    raw_sample = 'DAP2' if method == 'NMR2' else sample
                    _, a_data = pu.load_nmr_raw(raw_sample, f'{temp}C')
                    r_cell = f'{a_data.max():.2f}'
                    meth_prefix = (rf'\multirow{{{nmr_n}}}{{*}}{{NMR}} & '
                                   if is_first_row else r'~ & ')
                    is_first_row = False
                    temp_prefix = (rf'\multirow{{{len(methods_here)}}}{{*}}{{{temp}}} & '
                                   if ri == 0 else r'~ & ')
                    lines.append(f'        ~ & {meth_prefix}{temp_prefix}'
                                 f'{k1} & {k2} & {m} & {n} & {r_cell} \\\\')
        else:
            # Non-DAP: standard NMR block
            nmr_temps_present = [t for t in NMR_TEMPS if get_row(df, 'NMR', sample, t) is not None]
            is_first_row = True
            for ti, temp in enumerate(NMR_TEMPS):
                row = get_row(df, 'NMR', sample, temp)
                if row is None:
                    continue
                k1 = fmt_k(row['log_k1_median'], row['log_k1_CI_lower'],
                           row['log_k1_CI_upper'], -6)
                k2 = fmt_k(row['log_k2_median'], row['log_k2_CI_lower'],
                           row['log_k2_CI_upper'], -3)
                m      = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
                n      = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
                _, a_data = pu.load_nmr_raw(sample, f'{temp}C')
                r_cell = f'{a_data.max():.2f}'
                prefix = (rf'\multirow{{{len(nmr_temps_present)}}}{{*}}{{NMR}} & '
                          if is_first_row else r'~ & ')
                is_first_row = False
                lines.append(f'        ~ & {prefix}{temp} & {k1} & {k2} & {m} & {n} & {r_cell} \\\\')

        if si < len(SAMPLE_ORDER) - 1:
            lines.append(r'    \midrule')

    lines += [
        r'    \end{tabular}',
        r'\end{table}',
    ]

    out = os.path.join(pu._figures_dir(), 'table_km.tex')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {out}')
    pu.write_provenance(
        __file__,
        datasets=['Posterior summaries — samples: EDA, DAP, DAB; methods: DSC (25–100 °C), NMR (25–40 °C), NMR2/DAP2 (25–40 °C)'],
        source_paths=[pu.DSC_CSV, pu.NMR_CSV],
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=None,
                        help='Redirect all output here instead of figures/ (does not touch the originals).')
    args = parser.parse_args()
    if args.outdir:
        pu.set_output_dir(args.outdir)
    main()
