# Publication Figures — Implementation Plan
*Epoxy Kinetics 2025 | Generated 2026-04-24*

This document is written for two audiences:
- **Reviewer model**: check the plan for scientific accuracy and internal consistency before implementation begins. See the [Reviewer Checklist](#reviewer-checklist) section.
- **Implementer model**: write the code. See [Implementation Details](#implementation-details).

### Revisions from review (2026-04-24)

Implementer: these are resolved — do not re-introduce the original behaviors.

1. **Fig 1 CI band removed.** Median ODE curve only (R6 was impractical without MCMC chains).
2. **Fig 3 Ea annotations** use a single per-panel `slot` counter so DSC and NMR labels don't collide. Precision bumped from `.0f` to `.1f`.
3. **Fig 3** has a shared marker-shape legend in the left panel (sample identity stays in the coloured annotation text).
4. **Fig 3** dead `m_fit` endpoint-slope line removed.
5. **Font**: explicit `font.serif` fallback list so macOS picks Times New Roman.
6. **R9 corrected**: `\ln` is supported by matplotlib mathtext — use it directly.
7. **Table caption** explains DSC `r` is free per temperature (= $\alpha_\infty$), NMR `r = 2.0` fixed.

Unchanged and intentional: no `r` panel in Fig 2 (NMR has no `r`); NMR2 markers shown in Fig 3 but folded into NMR for the fit line; symmetric `\pm halfCI` in the table despite asymmetric log-space posteriors (matches existing table style).

---

## Scientific Background (both audiences)

This project fits the **Kamal-Malkin (KM) ODE** to epoxy cure kinetics measured by two methods:

```
dα/dt = (k1 + k2·α^m) · (1 − α)^(n/2) · (r − α)^(n/2)
```

- **α** = fractional conversion (0 → 1 for NMR; 0 → α_∞ < 1 for DSC below T_g,∞)
- **k1, k2** = rate constants (units: s⁻¹); stored in data as **log₁₀(k)** — always verify the base
- **m, n** = reaction order exponents (dimensionless, typically 0.3–2)
- **r** = stoichiometric/termination parameter:
  - **NMR**: r = 2.0, fixed by stoichiometry ([H₀]/[E₀]); NOT a fit parameter → no r column in NMR posterior CSV
  - **DSC**: r = α_∞ = max(α_data) ≈ 0.6–0.85; free parameter with tight prior; IS in DSC posterior CSV
- **Samples**: DGEBA epoxy with three diamine hardeners: EDA (2C), DAP (3C), DAB (4C)
  - DAP2 = second NMR replicate of DAP (2026 run); same compound, treat as NMR2 method for DAP
- **Temperatures**:
  - DSC: 25, 33, 50, 60, 80, 100 °C (6 isothermal temperatures)
  - NMR: 25, 33, 40 °C (3 isothermal temperatures)
- **Activation energy**: Arrhenius, Ea = −R × slope of (ln k vs 1/T); expect ~50–60 kJ/mol

---

## Data Sources

> **WARNING:** `posterior_summary_combined.csv` in the repo root is OUTDATED — NMR rows have zero-width CIs (LS point estimates). Do NOT use it.

| Purpose | File (relative to `epoxy_HPC/`) | Notes |
|---|---|---|
| DSC posteriors | `posterior_summary_DSC.csv` | 18 rows; has r column; MCMC 95% CIs |
| NMR posteriors | `posterior_summary_NMR_28Mar.csv` | 12 rows; no r column; MCMC 95% CIs |
| DSC raw α(t) | `epoxy_data_13Mar2026.mat` | MATLAB format; scipy.io.loadmat |
| NMR raw α(t) | `cpmg_fit_results/all_samples.csv` | columns: sample, temp, elapsed_min, alpha |
| ODE solver | `model_module.py` | reuse; do not reimplement |

### Posterior CSV schema

Both CSVs share this column set (NMR lacks r columns):
```
Label, log_k1_median, log_k1_CI_lower, log_k1_CI_upper,
       log_k2_median, log_k2_CI_lower, log_k2_CI_upper,
       m_median, m_CI_lower, m_CI_upper,
       n_median, n_CI_lower, n_CI_upper,
       [r_median, r_CI_lower, r_CI_upper,]   ← DSC only
       log_sigma_median, log_sigma_CI_lower, log_sigma_CI_upper
```

Label format: `DSC_EDA_60C`, `NMR_DAB_25C`, `NMR_DAP2_33C`

### NMR raw CSV schema
```
sample, temp, scan, timestamp, n_avg, A, A_err, T2, T2_err,
beta, beta_err, dropped, elapsed_min, alpha
```
- `elapsed_min`: time in minutes since t=0
- `alpha`: NaN in the first row (scan 1) — drop NaN rows before use
- `dropped`: boolean; rows where `dropped == True` were rejected by CPMG filter — exclude these

### MATLAB file structure

Unknown until runtime (scipy not yet run on this machine). The `load_dsc_raw()` function must:
1. Load the file
2. Print all top-level keys if the expected structure isn't found
3. Raise a descriptive error

Common MATLAB struct patterns for this type of data:
- Nested struct: `data['EDA'][0,0]['t60C']` etc.
- Simple dict: `data['alpha_EDA_60C']`
- Cell array: needs `data['field'][0][i]`

Hardcode the field access pattern once confirmed on first run.

---

## Reviewer Checklist

Before approving implementation, verify these scientific accuracy points:

### R1 — log base
Confirm that `log_k1_median` in both CSVs is **log₁₀**, not ln.
- Verification: `10^(log_k1_median)` should equal k1 in s⁻¹; compare with `cpmg_fit_results/km_results.csv` which has raw k1 values (least-squares, same ballpark). E.g., NMR_EDA_25C: log_k1 = −4.29 → k1 = 5.1×10⁻⁵ s⁻¹; km_results.csv says k1 = 5.07×10⁻⁵ ✓
- **Implication**: convert to ln k for Arrhenius plots by multiplying by ln(10) ≈ 2.303

### R2 — CI interpretation
The CIs are **95% credible intervals** (Bayesian MCMC, emcee), not standard errors.
- In plots: label error bars as "95% CI" in figure caption, not "±σ" or "±SE"
- In the LaTeX table: caption already says "95% credible intervals" — preserve this

### R3 — NMR r column
NMR posterior CSV has **no r column** because r=2.0 is fixed, not estimated.
- `load_posteriors()` must not crash on missing r columns for NMR rows
- In Fig 2 parameter plots: do NOT show r for NMR (r=2.0 fixed, not a posterior)
- In LaTeX table: NMR r column shows "2.0 (fixed)" with a footnote

### R4 — Unit conversions for the table
`log_k_median` is in log₁₀ of k in **s⁻¹**.
- k1 in table units 10⁻⁶ s⁻¹: `val = 10^(log_k1_median) × 10^6`
- CI half-width in linear space (asymmetric, but table shows symmetric ± for compactness):
  `halfCI = (10^CI_upper − 10^CI_lower) / 2 × 10^6`
- k2 in table units 10⁻³ s⁻¹: `val = 10^(log_k2_median) × 10^3`; same CI formula
- Cross-check: EDA_25C DSC → log_k1 = −4.446 → k1 = 3.58×10⁻⁵ s⁻¹ = 35.8 × 10⁻⁶ s⁻¹
  Existing table shows `35.80 ± 0.08` ✓

### R5 — DAP2 handling
DAP2 is the second NMR run of DAP (same compound). In the code:
- Remap: `Sample='DAP2'` → `Sample='DAP'`, `Method='NMR2'`
- Arrhenius fit for DAP NMR: combine DAP (3 pts) + DAP2/NMR2 (3 pts) = 6 points for the fit line
- Table: DAP2 is a separate sub-block under DAP with label "NMR (2026)" or "NMR2"

### R6 — ODE model for Fig 1 CI band — SIMPLIFIED
We do NOT have MCMC chains persisted, only median + CI bounds. A true credible band would require posterior samples. Envelope-from-bounds (solving at lo/lo and hi/hi) assumes perfect k1–k2 correlation and can mislead.

**Decision**: drop the CI band for Fig 1. Plot only the median ODE curve on top of the raw data. If a band is wanted later, regenerate MCMC and sample 50 chains.

**Implementer**: do not implement any CI envelope in `fig1_representative.py`. Median line + scatter only.

### R7 — dα/dt data derivation (Fig 1 right panel)
Raw α(t) data is noisy near the start. Savitzky-Golay smoothing before differentiation is standard.
- Recommend: `window_length=7, polyorder=3` as starting point; check residuals visually
- If data spacing is irregular (NMR timestamps), interpolate to uniform grid first before SG filter
- The model dα/dt is just the ODE RHS: `f(t, alpha_median(t), params_median)` — no smoothing needed for model

### R8 — Ea sign convention
`slope = dln(k)/d(1/T)` from Arrhenius: `ln k = -Ea/R · (1/T) + ln A`
- slope is **negative** (rate increases with T)
- `Ea = −slope × R / 1000` (in kJ/mol, R = 8.3145 J/mol·K)
- Expected range: 50–70 kJ/mol for epoxy/amine systems

### R9 — Matplotlib mathtext labels
`text.usetex: False` (no system LaTeX). Use matplotlib mathtext:
- `$\alpha$`, `$k_1$`, `$k_2$`, `$E_\mathrm{a}$`, `$1/T$`, `$\ln k_1$`
- `\ln` IS supported by mathtext; use it directly.
- Font fallback matters: set `'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']` so macOS resolves to Times, matching the manuscript. Plain `'font.family': 'serif'` alone defaults to DejaVu Serif on macOS.

---

## Implementation Details

### Directory layout

All scripts live in `epoxy_HPC/pub_figs/`. They import from `pub_utils.py` in the same directory. They access data via relative paths `../` (i.e., `epoxy_HPC/`).

```
epoxy_HPC/pub_figs/
├── plan.md               ← this file
├── pub_utils.py
├── fig1_representative.py
├── fig2_parameters.py
├── fig3_arrhenius.py
├── fig4_ea_bars.py
├── make_table_km.py
└── figures/
    ├── fig1_representative.pdf
    ├── fig1_representative.png
    ├── fig2_parameters.pdf
    ├── fig2_parameters.png
    ├── fig3_arrhenius.pdf
    ├── fig3_arrhenius.png
    ├── fig4_ea_bars.pdf
    ├── fig4_ea_bars.png
    ├── table_km.tex
    └── data_YYYYMMDD/
        ├── posterior_summary_DSC.csv
        ├── posterior_summary_NMR_28Mar.csv
        ├── epoxy_data_13Mar2026.mat
        └── all_samples.csv
```

---

### pub_utils.py

```python
import os, sys, shutil
from datetime import date
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Add parent dir to path for model_module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model_module

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.join(_HERE, '..')          # epoxy_HPC/
_FIGURES  = os.path.join(_HERE, 'figures')

DSC_CSV   = os.path.join(_REPO, 'posterior_summary_DSC.csv')
NMR_CSV   = os.path.join(_REPO, 'posterior_summary_NMR_28Mar.csv')
MAT_FILE  = os.path.join(_REPO, 'epoxy_data_13Mar2026.mat')
NMR_RAW   = os.path.join(_REPO, 'cpmg_fit_results', 'all_samples.csv')

# ── Style ──────────────────────────────────────────────────────────────────────
SAMPLE_COLOR  = {'EDA': '#0072B2', 'DAP': '#E69F00', 'DAB': '#009E73'}
METHOD_MARKER = {'DSC': 'o', 'NMR': 's', 'NMR2': 'D'}
METHOD_FILL   = {'DSC': True,  'NMR': False, 'NMR2': False}
METHOD_LS     = {'DSC': '--',  'NMR': ':',   'NMR2': '-.'}
FIG_DPI = 600

FIG_SIZE_2PANEL = (6.5, 3.0)
FIG_SIZE_2x2    = (6.5, 5.0)
FIG_SIZE_1x2    = (6.5, 3.5)
FIG_SIZE_BARS   = (5.0, 3.5)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':         10,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,   # screen preview; savefig uses FIG_DPI
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'text.usetex':       False,
})

R_GAS = 8.3145  # J / (mol·K)

# ── Data provenance snapshot ───────────────────────────────────────────────────
def snapshot_data():
    """Copy input data files to figures/data_YYYYMMDD/ once per day."""
    tag = date.today().strftime('%Y%m%d')
    dest = os.path.join(_FIGURES, f'data_{tag}')
    if os.path.isdir(dest):
        return
    os.makedirs(dest, exist_ok=True)
    for src, name in [
        (DSC_CSV,  'posterior_summary_DSC.csv'),
        (NMR_CSV,  'posterior_summary_NMR_28Mar.csv'),
        (MAT_FILE, 'epoxy_data_13Mar2026.mat'),
        (NMR_RAW,  'all_samples.csv'),
    ]:
        shutil.copy2(src, os.path.join(dest, name))

# ── Load posteriors ────────────────────────────────────────────────────────────
def _parse_label(label):
    parts = label.strip().split('_')
    # Format: METHOD_SAMPLE_TEMPC  (e.g. DSC_EDA_60C, NMR_DAP2_33C)
    method, sample, temp_str = parts[0], parts[1], parts[2]
    temp_c = int(temp_str.replace('C', '').replace('c', ''))
    return method, sample, temp_c

def load_posteriors():
    dsc = pd.read_csv(DSC_CSV)
    nmr = pd.read_csv(NMR_CSV)
    df  = pd.concat([dsc, nmr], ignore_index=True, sort=False)

    parsed = df['Label'].apply(_parse_label)
    df['Method'] = [p[0] for p in parsed]
    df['Sample'] = [p[1] for p in parsed]
    df['Temp_C'] = [p[2] for p in parsed]

    # Remap DAP2 → DAP sample, NMR2 method
    mask = df['Sample'] == 'DAP2'
    df.loc[mask, 'Sample'] = 'DAP'
    df.loc[mask, 'Method'] = 'NMR2'

    df['inv_T'] = 1.0 / (df['Temp_C'] + 273.15)

    ln10 = np.log(10.0)
    for p in ('k1', 'k2'):
        col = f'log_{p}'
        df[f'ln{p}']     = df[f'{col}_median'] * ln10
        df[f'ln{p}_err'] = (df[f'{col}_CI_upper'] - df[f'{col}_CI_lower']) / 2.0 * ln10

    for p in ('m', 'n'):
        df[f'{p}']     = df[f'{p}_median']
        df[f'{p}_err'] = (df[f'{p}_CI_upper'] - df[f'{p}_CI_lower']) / 2.0

    # r: DSC only; NMR rows will have NaN naturally
    if 'r_median' in df.columns:
        df['r']     = df['r_median']
        df['r_err'] = (df['r_CI_upper'] - df['r_CI_lower']) / 2.0

    return df

# ── Load raw data ──────────────────────────────────────────────────────────────
def load_nmr_raw(sample, temp_str):
    """Return (t_min, alpha) for one NMR sample/temperature."""
    df = pd.read_csv(NMR_RAW)
    mask = (df['sample'] == sample) & (df['temp'] == temp_str) & (~df['dropped'])
    sub  = df[mask].dropna(subset=['alpha']).copy()
    sub  = sub.sort_values('elapsed_min')
    return sub['elapsed_min'].to_numpy(), sub['alpha'].to_numpy()

def load_dsc_raw(sample, temp_str):
    """Return (t_min, alpha) for one DSC sample/temperature.
    
    IMPORTANT: The internal structure of epoxy_data_13Mar2026.mat is unknown
    until the file is opened for the first time. On first run, this function
    will print all top-level keys and raise an error — use that output to
    update the field access logic below.
    """
    from scipy.io import loadmat
    mat = loadmat(MAT_FILE)
    # Remove MATLAB metadata keys
    keys = [k for k in mat.keys() if not k.startswith('_')]
    # ── TODO: update field access after first run reveals structure ──
    # Placeholder: raise with diagnostic
    raise NotImplementedError(
        f"DSC MATLAB structure not yet mapped. Available keys: {keys}\n"
        "Update load_dsc_raw() in pub_utils.py with the correct field access."
    )

# ── KM ODE solve ──────────────────────────────────────────────────────────────
def solve_km(k1, k2, m, n, r, t_eval_min):
    """Solve KM ODE; t_eval_min in minutes → return alpha array."""
    t_sec = t_eval_min * 60.0
    params = dict(k1=k1, k2=k2, m=m, n=n, r=r)
    # model_module.solve_model signature: check model_module.py for exact API
    alpha = model_module.solve_model(params, t_sec)
    return alpha

# ── Activation energy ─────────────────────────────────────────────────────────
def compute_ea(df, param, method, sample):
    """Weighted linear Arrhenius fit for one (method, sample, param) group.
    
    param: 'lnk1' or 'lnk2'
    Returns (Ea_kJ, Ea_err_kJ). Ea is positive for activated processes.
    Raises ValueError if fewer than 2 data points.
    """
    mask = (df['Method'] == method) & (df['Sample'] == sample)
    sub  = df[mask].dropna(subset=[param, f'{param}_err']).copy()
    if len(sub) < 2:
        raise ValueError(f"Not enough data for Ea fit: {method} {sample} {param}")

    x    = sub['inv_T'].to_numpy()
    y    = sub[param].to_numpy()
    yerr = sub[f'{param}_err'].to_numpy()

    # Avoid zero or negative sigma
    yerr = np.clip(yerr, 1e-8, None)
    popt, pcov = curve_fit(lambda X, slope, intercept: slope * X + intercept,
                           x, y, sigma=yerr, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    Ea_kJ     = -popt[0] * R_GAS / 1000.0
    Ea_err_kJ =  perr[0] * R_GAS / 1000.0
    return Ea_kJ, Ea_err_kJ

# ── Save figure ───────────────────────────────────────────────────────────────
def savefig(fig, name, dpi=FIG_DPI):
    os.makedirs(_FIGURES, exist_ok=True)
    base = os.path.join(_FIGURES, name)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=dpi, bbox_inches='tight')
    print(f'Saved: {base}.pdf / .png')
```

---

### fig1_representative.py

```python
"""Two-panel representative cure figure: alpha(t) left, dα/dt right.

Usage:
    python fig1_representative.py [--dsc SAMPLE_TEMP] [--nmr SAMPLE_TEMP]
    python fig1_representative.py --dsc EDA_60C --nmr EDA_33C
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import pub_utils as pu

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DSC = ('EDA', '60C')
DEFAULT_NMR = ('EDA', '33C')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dsc', default='EDA_60C',
                   help='DSC dataset as SAMPLE_TEMP, e.g. EDA_60C')
    p.add_argument('--nmr', default='EDA_33C',
                   help='NMR dataset as SAMPLE_TEMP, e.g. EDA_33C')
    a = p.parse_args()
    dsc_s, dsc_t = a.dsc.split('_', 1)
    nmr_s, nmr_t = a.nmr.split('_', 1)
    return (dsc_s, dsc_t), (nmr_s, nmr_t)

def get_median_params(df, method, sample, temp_c):
    row = df[(df['Method'] == method) & (df['Sample'] == sample) &
             (df['Temp_C'] == temp_c)].iloc[0]
    lnk = lambda col: 10 ** row[col]  # back to linear from log10
    r   = row.get('r', 2.0) if not np.isnan(row.get('r', np.nan)) else 2.0
    return dict(
        k1=lnk('log_k1_median'), k2=lnk('log_k2_median'),
        m=row['m_median'], n=row['n_median'], r=r
    )

def main():
    pu.snapshot_data()
    (dsc_s, dsc_t), (nmr_s, nmr_t) = parse_args()
    dsc_tc = int(dsc_t.replace('C',''))
    nmr_tc = int(nmr_t.replace('C',''))

    df = pu.load_posteriors()

    # -- Load raw data --
    # DSC: needs load_dsc_raw() — uncomment once MATLAB structure is mapped
    # t_dsc, a_dsc = pu.load_dsc_raw(dsc_s, dsc_t)
    t_nmr, a_nmr = pu.load_nmr_raw(nmr_s, nmr_t)

    # -- KM model curves --
    p_nmr = get_median_params(df, 'NMR', nmr_s, nmr_tc)
    t_model = np.linspace(0, t_nmr.max(), 300)
    a_model = pu.solve_km(**p_nmr, t_eval_min=t_model)

    # -- dα/dt: model --
    # ODE RHS at each model point
    k1, k2, m, n, r = p_nmr['k1'], p_nmr['k2'], p_nmr['m'], p_nmr['n'], p_nmr['r']
    dt_model = (a_model >= 0) & (a_model < r)
    dadt_model = np.zeros_like(a_model)
    dadt_model[dt_model] = (
        (k1 + k2 * a_model[dt_model]**m)
        * (1 - a_model[dt_model])**(n/2)
        * (r - a_model[dt_model])**(n/2)
    ) * 60.0  # convert s⁻¹ → min⁻¹

    # -- dα/dt: data (SG filter) --
    # Interpolate to uniform grid if spacing is irregular
    t_uniform  = np.linspace(t_nmr[0], t_nmr[-1], len(t_nmr))
    a_interp   = np.interp(t_uniform, t_nmr, a_nmr)
    wl = min(7, len(a_interp) - (1 if len(a_interp) % 2 == 0 else 0))
    wl = wl if wl >= 5 else 5
    a_smooth   = savgol_filter(a_interp, window_length=wl, polyorder=3)
    dt_min     = np.diff(t_uniform).mean()
    dadt_data  = np.gradient(a_smooth, dt_min)

    # -- Plot --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pu.FIG_SIZE_2PANEL)
    fig.subplots_adjust(wspace=0.35)

    color = pu.SAMPLE_COLOR.get(nmr_s, 'black')

    ax1.scatter(t_nmr, a_nmr, s=18, marker='s', facecolors='none',
                edgecolors=color, label=f'NMR ({nmr_s}, {nmr_t})', zorder=3)
    ax1.plot(t_model, a_model, color=color, lw=1.5, label='KM fit')
    ax1.set_xlabel(r'time (min)')
    ax1.set_ylabel(r'conversion, $\alpha$')
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False)

    ax2.scatter(t_uniform, dadt_data, s=10, marker='s', facecolors='none',
                edgecolors=color, alpha=0.5, zorder=3)
    ax2.plot(t_model, dadt_model, color=color, lw=1.5)
    ax2.set_xlabel(r'time (min)')
    ax2.set_ylabel(r'd$\alpha$/d$t$ (min$^{-1}$)')

    pu.savefig(fig, 'fig1_representative')
    plt.close(fig)

if __name__ == '__main__':
    main()
```

**NOTE for implementer:**
- No CI envelope — median ODE curve + scatter only (R6).
- DSC raw data loading is blocked on the MATLAB structure being known. Write the NMR-only version first, then add DSC once `load_dsc_raw()` is mapped. The figure should show one DSC overlay panel and one NMR panel side-by-side once both are working — or show both on the same axes with different markers.
- SG filter `window_length=7, polyorder=3` is a starting point; if the derivative looks over-smoothed at early times on sparse NMR sampling, drop to `window_length=5` and re-run. Visually inspect, don't tune blindly.

---

### fig2_parameters.py

```python
"""2×2 parameter scatter: ln k1, ln k2, m, n vs 1/T."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import pub_utils as pu

PARAMS  = [('lnk1', r'$\ln k_1$'), ('lnk2', r'$\ln k_2$'),
           ('m',    r'$m$'),        ('n',    r'$n$')]
YLIMS   = {'lnk1': (-25, 0), 'lnk2': (-10, 0), 'm': (0, 3), 'n': (0, 3)}
METHODS = ['DSC', 'NMR', 'NMR2']
SAMPLES = ['EDA', 'DAP', 'DAB']

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()

    fig, axes = plt.subplots(2, 2, figsize=pu.FIG_SIZE_2x2, sharex=False)
    axes_flat = axes.flatten()

    legend_handles = []
    legend_labels  = []

    for idx, (param, label) in enumerate(PARAMS):
        ax = axes_flat[idx]
        for method in METHODS:
            for sample in SAMPLES:
                sub = df[(df['Method'] == method) & (df['Sample'] == sample)]
                if sub.empty:
                    continue
                x    = sub['inv_T'].to_numpy()
                y    = sub[param].to_numpy()
                yerr = sub.get(f'{param}_err', sub[param] * 0).to_numpy()

                color  = pu.SAMPLE_COLOR[sample]
                marker = pu.METHOD_MARKER[method]
                filled = pu.METHOD_FILL[method]
                mfc    = color if filled else 'none'

                eb = ax.errorbar(x, y, yerr=yerr,
                                 fmt=marker, color=color, mfc=mfc,
                                 capsize=3, ms=5, lw=0.8, elinewidth=0.8,
                                 label=f'{sample}/{method}')
                if idx == 0:
                    legend_handles.append(eb)
                    legend_labels.append(f'{sample} {method}')

        lo, hi = YLIMS.get(param, (None, None))
        if lo is not None:
            current = ax.get_ylim()
            ax.set_ylim(max(current[0], lo), min(current[1], hi))
        ax.set_xlabel(r'$1/T$ (K$^{-1}$)')
        ax.set_ylabel(label)

    # Shared legend in first panel only
    axes_flat[0].legend(legend_handles, legend_labels,
                        fontsize=7, frameon=False, ncol=2)

    fig.tight_layout()
    pu.savefig(fig, 'fig2_parameters')
    plt.close(fig)

if __name__ == '__main__':
    main()
```

---

### fig3_arrhenius.py

```python
"""Arrhenius plots: ln k1 (left) and ln k2 (right) vs 1/T."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import pub_utils as pu

SAMPLES     = ['EDA', 'DAP', 'DAB']
METHODS     = ['DSC', 'NMR']  # NMR2 is folded into NMR for DAP fit line
PARAMS      = [('lnk1', r'$\ln k_1$', (-25, 0)),
               ('lnk2', r'$\ln k_2$', (-10, 0))]

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()

    # For Arrhenius fitting, combine DAP NMR + DAP NMR2 into a single NMR group
    df_fit = df.copy()
    df_fit.loc[df_fit['Method'] == 'NMR2', 'Method'] = 'NMR'

    fig, axes = plt.subplots(1, 2, figsize=pu.FIG_SIZE_1x2)

    from scipy.optimize import curve_fit as _cf

    for j, (param, ylabel, (ylo, yhi)) in enumerate(PARAMS):
        ax = axes[j]

        # Scatter points (all three methods including NMR2 for DAP)
        for method in ['DSC', 'NMR', 'NMR2']:
            for sample in SAMPLES:
                sub = df[(df['Method'] == method) & (df['Sample'] == sample)]
                if sub.empty:
                    continue
                x    = sub['inv_T'].to_numpy()
                y    = sub[param].to_numpy()
                yerr = sub[f'{param}_err'].to_numpy()
                color  = pu.SAMPLE_COLOR[sample]
                marker = pu.METHOD_MARKER[method]
                mfc    = color if pu.METHOD_FILL[method] else 'none'
                ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color,
                            mfc=mfc, capsize=3, ms=5, lw=0.8, elinewidth=0.8)

        # Fit lines + Ea annotations. Use a SINGLE per-panel slot counter so
        # DSC and NMR annotations don't land at the same y-positions.
        slot = 0
        for method in METHODS:            # ['DSC', 'NMR']; NMR2 folded into NMR in df_fit
            for sample in SAMPLES:
                try:
                    Ea, Ea_err = pu.compute_ea(df_fit, param, method, sample)
                except ValueError:
                    continue
                sub = df_fit[(df_fit['Method'] == method) & (df_fit['Sample'] == sample)]
                x = sub['inv_T'].to_numpy()
                y = sub[param].to_numpy()
                x_fit = np.linspace(x.min(), x.max(), 100)
                popt, _ = _cf(lambda X, s, b: s*X + b, x, y)
                y_fit   = popt[0] * x_fit + popt[1]
                ax.plot(x_fit, y_fit, color=pu.SAMPLE_COLOR[sample],
                        ls=pu.METHOD_LS[method], lw=1.0, alpha=0.7)
                ax.text(0.98, 0.03 + 0.06 * slot,
                        f'{sample} {method}: {Ea:.1f}±{Ea_err:.1f} kJ/mol',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=7, color=pu.SAMPLE_COLOR[sample])
                slot += 1

        ax.set_xlabel(r'$1/T$ (K$^{-1}$)')
        ax.set_ylabel(ylabel)
        current = ax.get_ylim()
        ax.set_ylim(max(current[0], ylo), min(current[1], yhi))

    # Shared marker legend (method identity only; sample identity is in the
    # coloured annotation text). Place in first panel.
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0],[0], marker='o', color='k', mfc='k',   ls='', ms=5, label='DSC'),
        Line2D([0],[0], marker='s', color='k', mfc='none', ls='', ms=5, label='NMR'),
        Line2D([0],[0], marker='D', color='k', mfc='none', ls='', ms=5, label='NMR (2026)'),
    ]
    axes[0].legend(handles=marker_handles, loc='upper left',
                   frameon=False, fontsize=7)

    fig.tight_layout()
    pu.savefig(fig, 'fig3_arrhenius')
    plt.close(fig)

if __name__ == '__main__':
    main()
```

---

### fig4_ea_bars.py

```python
"""Grouped bar chart: Ea for k1 (left) and k2 (right)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import pub_utils as pu

SAMPLES = ['EDA', 'DAP', 'DAB']
METHODS = ['DSC', 'NMR']
METHOD_BAR_COLOR = {'DSC': '#0072B2', 'NMR': '#D55E00'}

def main():
    pu.snapshot_data()
    df = pu.load_posteriors()
    df_fit = df.copy()
    df_fit.loc[df_fit['Method'] == 'NMR2', 'Method'] = 'NMR'

    fig, axes = plt.subplots(1, 2, figsize=pu.FIG_SIZE_BARS)
    x = np.arange(len(SAMPLES))
    width = 0.35

    for j, (param, title) in enumerate([('lnk1', r'$k_1$'), ('lnk2', r'$k_2$')]):
        ax = axes[j]
        for i, method in enumerate(METHODS):
            Eas, errs = [], []
            for sample in SAMPLES:
                try:
                    Ea, err = pu.compute_ea(df_fit, param, method, sample)
                except ValueError:
                    Ea, err = 0.0, 0.0
                Eas.append(Ea)
                errs.append(err)
            offset = (i - 0.5) * width
            ax.bar(x + offset, Eas, width, yerr=errs,
                   label=method, color=METHOD_BAR_COLOR[method],
                   capsize=4, error_kw=dict(lw=1.0), zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(SAMPLES)
        ax.set_ylabel(r'$E_\mathrm{a}$ (kJ mol$^{-1}$)')
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.set_ylim(0, None)

    fig.tight_layout()
    pu.savefig(fig, 'fig4_ea_bars')
    plt.close(fig)

if __name__ == '__main__':
    main()
```

---

### make_table_km.py

```python
"""Generate LaTeX KM parameter table for SI.

Matches structure of Epoxy-Kinetics-2025/SI/KM_with_r_table.tex.
NMR r = 2.0 fixed (not fit); DSC r from posterior median.
DAP2 shown as NMR2 sub-block under DAP.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pub_utils as pu

SAMPLE_ORDER = ['EDA', 'DAP', 'DAB']
DSC_TEMPS    = [25, 33, 50, 60, 80, 100]
NMR_TEMPS    = [25, 33, 40]

def fmt_val(median, ci_lo, ci_hi, scale=1.0):
    """Format median ± halfCI in linear space, scaled."""
    val      = median * scale
    half_ci  = abs(ci_hi - ci_lo) / 2.0 * scale
    # Determine decimal places from half_ci magnitude
    if half_ci == 0:
        return rf'\num{{{val:.3g}}}'
    import math
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
    import math
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
        n_nmr2 = 3 if sample == 'DAP' else 0
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

            prefix = rf'\multirow{{{n_total}}}{{*}}{{{sample}}} & ' \
                     rf'\multirow{{6}}{{*}}{{DSC}} & ' if ti == 0 \
                else r'~ & ~ & '
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
            m  = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
            n  = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
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
                m  = fmt_val(row['m_median'], row['m_CI_lower'], row['m_CI_upper'])
                n  = fmt_val(row['n_median'], row['n_CI_lower'], row['n_CI_upper'])
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

if __name__ == '__main__':
    main()
```

---

## Known Blockers

1. **MATLAB structure** — `load_dsc_raw()` will raise `NotImplementedError` on first run and print available keys. Update the field access logic in `pub_utils.py` after seeing the keys.

2. **model_module.py API** — The exact signature of `solve_model` in `model_module.py` is assumed to be `solve_model(params_dict, t_seconds)`. Check the actual signature before calling and adjust `solve_km()` accordingly.

3. **scipy availability** — If `scipy` is not in the active Python environment, install it: `pip install scipy` or `conda install scipy`.

---

## Verification Checklist

```bash
cd epoxy_HPC/pub_figs
python fig2_parameters.py     # safest first: no MATLAB needed
python fig3_arrhenius.py
python fig4_ea_bars.py
python make_table_km.py
# After MATLAB structure is mapped:
python fig1_representative.py --dsc EDA_60C --nmr EDA_33C
```

- [ ] `figures/` has 10 PDF/PNG + table_km.tex + data_YYYYMMDD/ folder
- [ ] Fig 2: DSC points have error bars; NMR points have open markers and error bars
- [ ] Fig 3: Ea annotations in range 40–80 kJ/mol; fit lines pass through data
- [ ] Fig 4: DSC bars ≠ NMR bars for most samples; bars are positive
- [ ] table_km.tex: `pdflatex` compiles with `\usepackage{booktabs,multirow,siunitx,pdflscape}`
- [ ] Cross-check EDA 25°C DSC k1: should be ~35.8 × 10⁻⁶ s⁻¹ (from existing table)
