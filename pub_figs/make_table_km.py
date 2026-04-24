"""Generate LaTeX KM parameter table for SI.

Matches structure of Epoxy-Kinetics-2025/SI/KM_with_r_table.tex.
NMR r = 2.0 fixed (not fit); DSC r from posterior median.
DAP2 shown as NMR2 sub-block under DAP.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pub_utils as pu

SAMPLE_ORDER = ['EDA', 'DAP', 'DAB']
DSC_TEMPS    = [25, 33, 50, 60, 80, 100]
NMR_TEMPS    = [25, 33, 40]

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
        r'\begin{landscape}',
        r'\begin{table}[!ht]',
        r'\caption{Modified Kamal-Malkin fits of DGEBA with '
        r'ethylenediamine (EDA), 1,3-diaminopropane (DAP), and '
        r'1,4-diaminobutane (DAB). Reported values are posterior medians; '
        r'uncertainties represent the 95\% credible intervals from the '
        r'Bayesian analysis. For DSC, $r$ is a free parameter per temperature '
        r'(equal to $\alpha_\infty$); for NMR, $r = 2.0$ is fixed by '
        r'stoichiometry.$^\dagger$}',
        r'\label{tab:KM_with_r}',
        r'    \centering',
        r'    \begin{tabular}{c|cc ccccc}',
        r'        Sample & Method & Temp ($^\circ$C) & '
        r'$k_1/10^{-6}$ [s$^{-1}$] & $k_2/10^{-3}$ [s$^{-1}$] & '
        r'$m$ & $n$ & $r$ \\ \midrule',
    ]

    for si, sample in enumerate(SAMPLE_ORDER):
        # Count total rows: 6 DSC + 3 NMR + (3 NMR2 if DAP)
        n_nmr2  = 3 if sample == 'DAP' else 0
        n_total = 6 + 3 + n_nmr2

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

        # NMR rows (original run, Method='NMR')
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
            r_cell = r'\num{2.0}$^\ddagger$'

            prefix = rf'\multirow{{3}}{{*}}{{NMR}} & ' if ti == 0 else r'~ & '
            lines.append(f'        ~ & {prefix}{temp} & {k1} & {k2} & {m} & {n} & {r_cell} \\\\')

        # NMR2 (DAP2 → DAP only)
        if sample == 'DAP':
            lines.append(r'        \cline{2-8}')
            for ti, temp in enumerate(NMR_TEMPS):
                row = get_row(df, 'NMR2', sample, temp)
                if row is None:
                    continue
                k1 = fmt_k(row['log_k1_median'], row['log_k1_CI_lower'],
                           row['log_k1_CI_upper'], -6)
                k2 = fmt_k(row['log_k2_median'], row['log_k2_CI_lower'],
                           row['log_k2_CI_upper'], -3)
                m      = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
                n      = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
                r_cell = r'\num{2.0}$^\ddagger$'
                prefix = rf'\multirow{{3}}{{*}}{{NMR (2026)}} & ' if ti == 0 else r'~ & '
                lines.append(f'        ~ & {prefix}{temp} & {k1} & {k2} & {m} & {n} & {r_cell} \\\\')

        if si < len(SAMPLE_ORDER) - 1:
            lines.append(r'    \midrule')

    lines += [
        r'    \end{tabular}',
        r'    \footnotesize{$^\dagger$NMR fits use $r = 2.0$ fixed by '
        r'stoichiometry ($[\mathrm{H}_0]/[\mathrm{E}_0]$) and are not '
        r'estimated from the data.}\\',
        r'    \footnotesize{$^\ddagger$See text; \num{2.0} (fixed).}',
        r'\end{table}',
        r'\end{landscape}',
    ]

    out = os.path.join(os.path.dirname(__file__), 'figures', 'table_km.tex')
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
    main()
