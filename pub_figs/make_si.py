"""Generate all SI figures and the master si_figures.tex.

Usage
-----
From the repo root (epoxy_HPC/):

    conda run -n python_coding python pub_figs/make_si.py

Or from pub_figs/:

    conda run -n python_coding python make_si.py

What it produces
----------------
  figures/SI_figures/cpmg_fit_{sample}_{temp}.pdf/.png   (12 figures)
  figures/SI_figures/rep_DSC_{sample}_{temp}.pdf/.png    (18 figures)
  figures/SI_figures/rep_NMR_{sample}_{temp}.pdf/.png     (9 figures)
  figures/SI_figures/rep_NMR2_{sample}_{temp}.pdf/.png    (3 replicate figures)
  figures/si_figures.tex                                 (master SI LaTeX, ready to \\input)
  figures/make_si.provenance.txt                         (MD5/mtime of all source files)

The si_figures.tex file uses absolute paths for \\includegraphics and ends with
\\input{...table_km.tex} (also absolute). To include it in your main document:

    \\input{/full/path/to/pub_figs/figures/si_figures.tex}

Re-run this script any time the posteriors or raw data change.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pub_utils as pu
from cpmg_fit      import make_cpmg_figure
from representative import make_rep_figure, get_param_bounds

# ── Dataset tables ─────────────────────────────────────────────────────────────
DSC_TEMPS  = [25, 33, 50, 60, 80, 100]
NMR_TEMPS  = [25, 33, 40, 60]
NMR2_TEMPS = [25, 33, 40]   # DAP2 replicate was never re-run at 60C
SAMPLES    = ['EDA', 'DAP', 'DAB']

DSC_SETS  = [(s, t) for s in SAMPLES for t in DSC_TEMPS]
NMR_SETS  = [(s, t) for s in SAMPLES for t in NMR_TEMPS]
NMR2_SETS = [('DAP', t) for t in NMR2_TEMPS]   # DAP2 replicate; Sample='DAP' in posteriors
CPMG_SETS = [(s, t, False) for s, t in NMR_SETS] + [('DAP', t, True) for t in NMR2_TEMPS]

FULL_NAMES = {
    'EDA': 'ethylenediamine (EDA)',
    'DAP': '1,3-diaminopropane (DAP)',
    'DAB': '1,4-diaminobutane (DAB)',
}

# ── Caption builders ───────────────────────────────────────────────────────────
def _cap_cpmg(sample, temp, replicate=False):
    rep = r' (replicate experiment)' if replicate else ''
    return (
        rf'CPMG stretched-exponential fit parameters for DGEBA/{FULL_NAMES[sample]} '
        rf'at {temp}\,\textdegree C{rep}. '
        r'Top: transverse relaxation time $T_2$ vs.\ elapsed cure time. '
        r'Bottom: stretching exponent $\beta$ vs.\ elapsed cure time. '
        r'Error bars are fit uncertainties.'
    )

def _cap_rep(method, sample, temp, replicate=False):
    rep = r' (replicate experiment)' if replicate else ''
    return (
        rf'Kamal-Malkin model fit to {method} cure data for '
        rf'DGEBA/{FULL_NAMES[sample]} at {temp}\,\textdegree C{rep}. '
        r'Left: conversion $\alpha(t)$; shaded band is the 95\% credible interval. '
        r'Right: reaction rate $\mathrm{d}\alpha(t)/\mathrm{d}t$ vs.\ $\alpha$.'
    )

# ── LaTeX figure environment builder ──────────────────────────────────────────
def _fig_env(stem, caption, label):
    abspath = os.path.join(pu._figures_dir(), stem) + '.pdf'
    return '\n'.join([
        r'\begin{figure}[!ht]',
        r'  \centering',
        rf'  \includegraphics[width=0.5\linewidth]{{{abspath}}}',
        rf'  \caption{{{caption}}}',
        rf'  \label{{{label}}}',
        r'\end{figure}',
    ])

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    pu.snapshot_data()
    df = pu.load_posteriors()

    si_dir = os.path.join(pu._figures_dir(), 'SI_figures')
    os.makedirs(si_dir, exist_ok=True)

    tex_blocks = []

    # ── Section 1: CPMG relaxation parameters ─────────────────────────────────
    tex_blocks.append(r'\clearpage')
    tex_blocks.append(r'\section{CPMG Relaxation Parameters}')
    tex_blocks.append('')

    for sample, temp, replicate in CPMG_SETS:
        temp_str = f'{temp}C'
        # Raw data uses 'DAP2' for the replicate; posteriors use 'DAP'/'NMR2'
        raw_sample = 'DAP2' if replicate else sample
        stem = make_cpmg_figure(raw_sample, temp_str)
        label = f'fig:cpmg_{sample}_{temp_str}' + ('_rep' if replicate else '')
        cap   = _cap_cpmg(sample, temp, replicate)
        tex_blocks.append(_fig_env(stem, cap, label))
        tex_blocks.append('')

    # ── Section 2: Representative DSC fits ────────────────────────────────────
    tex_blocks.append(r'\clearpage')
    tex_blocks.append(r'\section{Representative DSC Fits}')
    tex_blocks.append('')

    for sample, temp in DSC_SETS:
        temp_str = f'{temp}C'
        stem  = make_rep_figure(df, sample, temp_str, temp, 'DSC')
        label = f'fig:rep_DSC_{sample}_{temp_str}'
        cap   = _cap_rep('DSC', sample, temp)
        tex_blocks.append(_fig_env(stem, cap, label))
        tex_blocks.append('')

    # ── Section 3: Representative NMR fits ────────────────────────────────────
    tex_blocks.append(r'\clearpage')
    tex_blocks.append(r'\section{Representative NMR Fits}')
    tex_blocks.append('')

    for sample, temp in NMR_SETS:
        temp_str = f'{temp}C'
        stem  = make_rep_figure(df, sample, temp_str, temp, 'NMR')
        label = f'fig:rep_NMR_{sample}_{temp_str}'
        cap   = _cap_rep('NMR', sample, temp)
        tex_blocks.append(_fig_env(stem, cap, label))
        tex_blocks.append('')

    for sample, temp in NMR2_SETS:
        temp_str = f'{temp}C'
        stem  = make_rep_figure(df, sample, temp_str, temp, 'NMR2')
        label = f'fig:rep_NMR_{sample}_{temp_str}_rep'
        cap   = _cap_rep('NMR', sample, temp, replicate=True)
        tex_blocks.append(_fig_env(stem, cap, label))
        tex_blocks.append('')

    # ── Section 4: KM parameter table ─────────────────────────────────────────
    tex_blocks.append(r'\clearpage')
    tex_blocks.append(r'\section{Kamal-Malkin Fit Parameters}')
    tex_blocks.append('')
    table_km_abs = os.path.join(pu._figures_dir(), 'table_km.tex')
    tex_blocks.append(rf'\input{{{table_km_abs}}}')
    tex_blocks.append('')

    # ── Write si_figures.tex ───────────────────────────────────────────────────
    out = os.path.join(pu._figures_dir(), 'si_figures.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(tex_blocks) + '\n')
    print(f'Saved: {out}')

    pu.write_provenance(
        __file__,
        datasets=[
            'NMR raw CPMG fit results — EDA/DAP/DAB/DAP2 at 25, 33, 40 °C',
            'Posterior summaries — DSC: EDA/DAP/DAB at 25–100 °C; NMR: EDA/DAP/DAB/DAP2 at 25–40 °C',
        ],
        source_paths=[pu.NMR_RAW, pu.DSC_CSV, pu.NMR_CSV, pu.MAT_FILE],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=None,
                        help='Redirect all output here instead of figures/ (does not touch the originals).')
    args = parser.parse_args()
    if args.outdir:
        pu.set_output_dir(args.outdir)
    main()
