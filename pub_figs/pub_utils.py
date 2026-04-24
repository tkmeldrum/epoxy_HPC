import os, sys, shutil, hashlib
from datetime import date, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

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

_DSC_TEMP_INDEX = {25: 0, 33: 1, 50: 2, 60: 3, 80: 4, 100: 5}

def load_dsc_raw(sample, temp_str):
    """Return (t_min, alpha, dadt_per_min) for one DSC sample/temperature.

    Uses clean_alpha_unscaled (true fractional conversion, 0 → α_∞ = r).
    clean_dadt is d(clean_alpha_normalised)/dt, so true dadt = clean_dadt * r.
    Temperatures map to indices: 25→0, 33→1, 50→2, 60→3, 80→4, 100→5
    """
    from scipy.io import loadmat
    temp_c = int(str(temp_str).replace('C', '').replace('c', ''))
    if temp_c not in _DSC_TEMP_INDEX:
        raise ValueError(f"Unknown DSC temperature {temp_str}; valid: {list(_DSC_TEMP_INDEX)}")
    idx  = _DSC_TEMP_INDEX[temp_c]
    mat  = loadmat(MAT_FILE)
    data = mat[sample][0, 0]
    t_sec  = data['clean_time'][0, idx].flatten()
    alpha  = data['clean_alpha_unscaled'][0, idx].flatten()
    r      = alpha.max()
    dadt_s = data['clean_dadt'][0, idx].flatten() * r   # normalised → true s⁻¹
    # dadt is one point shorter (finite difference); trim t to match
    n      = min(len(t_sec), len(dadt_s))
    return t_sec[:n] / 60.0, alpha[:n], dadt_s[:n] * 60.0   # min, –, min⁻¹

# ── KM ODE solve ──────────────────────────────────────────────────────────────
def solve_km(k1, k2, m, n, r, t_eval_min, a0=1e-10):
    """Solve KM ODE; t_eval_min in minutes → return alpha array.

    k1, k2 are linear rate constants (s⁻¹); converted to log10 internally.
    Uses a local RHS wrapper that clamps (1−α) ≥ 0 before the fractional
    power, preventing NaN when α overshoots 1.0 during integration (NMR r=2).
    """
    log_k1 = np.log10(k1)
    log_k2 = np.log10(k2)
    t_sec  = t_eval_min * 60.0

    def _rhs(t, a):
        a_c  = np.clip(a, 1e-10, r - 1e-10)
        epox = np.maximum(0.0, 1.0 - a_c)    # zero rate once α ≥ 1
        return ((10**log_k1 + 10**log_k2 * a_c**m)
                * epox**(n / 2)
                * (r - a_c)**(n / 2))

    sol = solve_ivp(
        _rhs,
        [t_sec[0], t_sec[-1]], [a0],
        t_eval=t_sec,
        method='LSODA', rtol=1e-6, atol=1e-8,
    )
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return np.full_like(t_sec, np.nan)
    return sol.y[0]

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

# ── Provenance ────────────────────────────────────────────────────────────────
def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write_provenance(script_path, datasets, source_paths):
    """Write figures/<script_stem>.provenance.txt with dataset and file metadata."""
    os.makedirs(_FIGURES, exist_ok=True)
    stem = os.path.splitext(os.path.basename(script_path))[0]
    out  = os.path.join(_FIGURES, f'{stem}.provenance.txt')
    now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        'PROVENANCE RECORD',
        f'Generated : {now}',
        f'Script    : {os.path.abspath(script_path)}',
        '',
        'DATASETS USED',
    ]
    for d in (datasets if isinstance(datasets, list) else [datasets]):
        lines.append(f'  {d}')
    lines += ['', 'SOURCE FILES']
    for p in source_paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S')
            size  = os.path.getsize(p)
            md5   = _md5(p)
        else:
            mtime, size, md5 = 'FILE NOT FOUND', '', ''
        lines += [
            f'  {os.path.basename(p)}',
            f'    path    : {p}',
            f'    modified: {mtime}',
            f'    size    : {size} bytes' if size != '' else '    size    : —',
            f'    md5     : {md5}',
        ]
    lines.append('')

    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Provenance: {out}')

# ── Save figure ───────────────────────────────────────────────────────────────
def savefig(fig, name, dpi=FIG_DPI):
    os.makedirs(_FIGURES, exist_ok=True)
    base = os.path.join(_FIGURES, name)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    fig.savefig(base + '.png', dpi=dpi, bbox_inches='tight')
    print(f'Saved: {base}.pdf / .png')
