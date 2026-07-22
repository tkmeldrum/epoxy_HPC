"""CPMG stretched-exponential fit parameters: T₂ and β vs elapsed time.

Shows how the NMR relaxation time (T₂) and stretching exponent (β) evolve
as the epoxy cures — the underlying observables from which α is derived.

Default: EDA 25 °C  (change SAMPLE / TEMP_STR to select another dataset)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import io
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pub_utils as pu
import local_config

SAMPLE   = 'EDA'
TEMP_STR = '25C'

T2_LIM   = (1e-5, 1e-1)   # log-scale T₂ axis (s)
BETA_LIM = (0.0,  2.5)    # linear β axis

COLOR_FIRST = '#56B4E9'   # sky blue  — first time point
COLOR_LAST  = '#D55E00'   # vermillion — last time point

# Mirrors EXPERIMENTS / DAP2_* in cpmg_batch_fit.py
EXPERIMENTS = {
    "25C": {"EDA": "WM39", "DAP": "WM38", "DAB": "WM40"},
    "33C": {"EDA": "WM66", "DAP": "WM63", "DAB": "WM65"},
    "40C": {"EDA": "WM58", "DAP": "WM54", "DAB": "WM53"},
}
_ZIP_ROOT = Path(local_config.DATA_ROOT)
DAP2_ZIP  = _ZIP_ROOT / "DAP2.zip"
DAP2_EXPERIMENTS = {
    "25C": "Epoxy2026/13DAP/CPMG_25C",
    "33C": "Epoxy2026/13DAP/CPMG_33C",
    "40C": "Epoxy2026/13DAP/CPMG_40C_2",
}

# 60C: new temperature, one zip per sample (mirrors cpmg_batch_fit.py)
EXPERIMENTS_60C = {
    # EDA's first attempt ("debugger") was superseded by a repeat run
    # ("debugger4") -- see README for why.
    "EDA": {"zip": _ZIP_ROOT / "DGEBA_EDA_60.zip", "prefix": "debugger4"},
    "DAP": {"zip": _ZIP_ROOT / "DGEBA_DAP_60.zip", "prefix": "debugger"},
    "DAB": {"zip": _ZIP_ROOT / "DGEBA_DAB_60.zip", "prefix": "debugger"},
}


# ── Kea binary readers (ported from cpmg_batch_fit.py) ───────────────────────

def _parse_value(val_str):
    val_str = val_str.strip()
    if val_str.startswith("'") and val_str.endswith("'"):
        return val_str[1:-1]
    if val_str.startswith("[") and val_str.endswith("]"):
        parts = re.split(r"[\s,]+", val_str[1:-1].strip())
        return [float(p) for p in parts if p]
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        return val_str


def _read_params_from_bytes(raw_bytes: bytes) -> dict:
    params = {}
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if key[0] == "9":
            params["amp90"] = _parse_value(val)
        elif key[0] == "1":
            params["amp180"] = _parse_value(val)
        else:
            try:
                params[key] = _parse_value(val)
            except Exception:
                params[key] = val
    return params


def _autophase(d):
    if not np.iscomplexobj(d):
        return d
    theta = -np.angle(d.sum())
    return d * np.exp(1j * theta)


def _prepare_t2_from_bytes(data_bytes: bytes, params: dict):
    """Parse Kea binary data from bytes. Returns (echo_vec_s, intdata)."""
    f = io.BytesIO(data_bytes)
    f.read(4); f.read(4); f.read(4)
    data_type = struct.unpack("<i", f.read(4))[0]
    x_dim     = struct.unpack("<i", f.read(4))[0]
    y_dim     = struct.unpack("<i", f.read(4))[0]
    f.read(4); f.read(4)

    total = x_dim * y_dim

    if data_type == 500:
        data = np.frombuffer(f.read(total * 4), dtype="<f4").copy()
    elif data_type == 501:
        raw  = np.frombuffer(f.read(total * 8), dtype="<f4")
        data = (raw[0::2] + 1j * raw[1::2]).copy()
    elif data_type == 502:
        data = np.frombuffer(f.read(total * 8), dtype="<f8").copy()
    elif data_type == 503:
        raw  = np.frombuffer(f.read(total * 2 * 4), dtype="<f4")
        data = np.column_stack([raw[:total], raw[total:]])
    elif data_type == 504:
        raw  = np.frombuffer(f.read(total * 3 * 4), dtype="<f4").reshape(-1, 3)
        cpx  = raw[:x_dim, 1:3]
        data = (cpx[:, 0] + 1j * cpx[:, 1]).copy()
        y_dim = x_dim
        x_dim = 1
    else:
        raise ValueError(f"Unknown Kea dataType: {data_type}")

    echo_time = float(params["echoTime"])  # µs

    if y_dim > 1:
        timedata = data[: x_dim * y_dim].reshape((x_dim, y_dim), order="F")
        echo_vec_cpx = timedata.sum(axis=0)
        intdata  = np.real(_autophase(echo_vec_cpx))
        nr_echoes = y_dim
    else:
        intdata   = np.real(_autophase(data)) if np.iscomplexobj(data) else np.asarray(data, dtype=float)
        nr_echoes = x_dim

    echo_vec = 1e-6 * echo_time * np.arange(1, nr_echoes + 1)
    return echo_vec, intdata


def _list_scans(zf: zipfile.ZipFile, wm_id: str) -> list[dict]:
    prefix = f"{wm_id}/Debugger/CPMG/"
    scans = {}
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        parts = remainder.split("/")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        fname = parts[1]
        if idx not in scans:
            scans[idx] = {"index": idx, "data_info": None, "par_info": None}
        if fname == "data.2d":
            scans[idx]["data_info"] = info
        elif fname == "acqu.par":
            scans[idx]["par_info"] = info
    complete = [s for s in scans.values() if s["data_info"] and s["par_info"]]
    complete.sort(key=lambda s: s["index"])
    return complete


def _list_scans_direct(zf: zipfile.ZipFile, prefix: str) -> list[dict]:
    prefix = prefix.rstrip("/") + "/"
    scans = {}
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix):]
        parts = remainder.split("/")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        fname = parts[1]
        if idx not in scans:
            scans[idx] = {"index": idx, "data_info": None, "par_info": None}
        if fname == "data.2d":
            scans[idx]["data_info"] = info
        elif fname == "acqu.par":
            scans[idx]["par_info"] = info
    complete = [s for s in scans.values() if s["data_info"] and s["par_info"]]
    complete.sort(key=lambda s: s["index"])
    return complete


# ── Scan loading ──────────────────────────────────────────────────────────────

def _load_scan_decay(sample, temp_str, scan_num, n_avg):
    """Load raw CPMG echo decay for one scan (or averaged group).

    Returns (t_echo_s, y_raw) where y_raw is in original instrument units.
    Raises FileNotFoundError if the zip is not accessible.
    """
    if sample == 'DAP2':
        zip_path = DAP2_ZIP
        with zipfile.ZipFile(zip_path) as zf:
            all_scans = _list_scans_direct(zf, DAP2_EXPERIMENTS[temp_str])
            start_pos = next((i for i, s in enumerate(all_scans)
                              if s['index'] == scan_num), None)
            if start_pos is None:
                raise ValueError(f"Scan {scan_num} not found for {sample} {temp_str}")
            group = all_scans[start_pos: start_pos + max(1, n_avg)]
            t_ref = None
            ys = []
            for s in group:
                par_bytes  = zf.read(s['par_info'].filename)
                data_bytes = zf.read(s['data_info'].filename)
                params = _read_params_from_bytes(par_bytes)
                t, y   = _prepare_t2_from_bytes(data_bytes, params)
                if t_ref is None:
                    t_ref = t
                ys.append(y)
    elif sample in EXPERIMENTS_60C:
        info = EXPERIMENTS_60C[sample]
        zip_path = info["zip"]
        with zipfile.ZipFile(zip_path) as zf:
            all_scans = _list_scans_direct(zf, info["prefix"])
            start_pos = next((i for i, s in enumerate(all_scans)
                              if s['index'] == scan_num), None)
            if start_pos is None:
                raise ValueError(f"Scan {scan_num} not found for {sample} {temp_str}")
            group = all_scans[start_pos: start_pos + max(1, n_avg)]
            t_ref = None
            ys = []
            for s in group:
                par_bytes  = zf.read(s['par_info'].filename)
                data_bytes = zf.read(s['data_info'].filename)
                params = _read_params_from_bytes(par_bytes)
                t, y   = _prepare_t2_from_bytes(data_bytes, params)
                if t_ref is None:
                    t_ref = t
                ys.append(y)
    else:
        zip_path = _ZIP_ROOT / f'{temp_str}.zip'
        wm_id    = EXPERIMENTS[temp_str][sample]
        with zipfile.ZipFile(zip_path) as zf:
            all_scans = _list_scans(zf, wm_id)
            start_pos = next((i for i, s in enumerate(all_scans)
                              if s['index'] == scan_num), None)
            if start_pos is None:
                raise ValueError(f"Scan {scan_num} not found for {sample} {temp_str}")
            group = all_scans[start_pos: start_pos + max(1, n_avg)]
            t_ref = None
            ys = []
            for s in group:
                par_bytes  = zf.read(s['par_info'].filename)
                data_bytes = zf.read(s['data_info'].filename)
                params = _read_params_from_bytes(par_bytes)
                t, y   = _prepare_t2_from_bytes(data_bytes, params)
                if t_ref is None:
                    t_ref = t
                ys.append(y)

    n = min(len(y) for y in ys)
    y_avg = np.mean([y[:n] for y in ys], axis=0)
    return t_ref[:n], y_avg


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _cap_err(vals, errs, max_relative=0.2):
    """Cap error bars at max_relative * |value| to keep plots legible."""
    return np.minimum(errs, max_relative * np.abs(vals))


def _asterisks(ax, t_all, y_all, ylo, yhi, color='k'):
    """Plot ▲/▼ asterisk markers at axis boundaries for out-of-range points."""
    hi = y_all > yhi
    lo = y_all < ylo
    if hi.any():
        ax.scatter(t_all[hi], np.full(hi.sum(), yhi * 0.97),
                   marker='*', color=color, s=50, zorder=5, clip_on=False)
    if lo.any():
        ax.scatter(t_all[lo], np.full(lo.sum(), ylo * 1.05),
                   marker='*', color=color, s=50, zorder=5, clip_on=False)


def _plot_decay_panel(ax, t_echo, y_raw, A, T2, beta, color, elapsed_min):
    """Plot a single CPMG echo decay with stretched exponential fit overlaid."""
    t_ms    = t_echo * 1e3
    y_norm  = y_raw / A if A > 0 else y_raw / np.max(np.abs(y_raw))
    ax.scatter(t_ms, y_norm, s=3, color='0.65', zorder=2, linewidths=0)

    t_fine = np.linspace(t_echo[0], t_echo[-1], 500)
    y_fit  = np.exp(-(t_fine / T2) ** beta)
    ax.plot(t_fine * 1e3, y_fit, color=color, lw=1.5, zorder=3)

    ax.set_xlabel('Echo center time (ms)')
    ax.set_ylabel('Normalised intensity')
    ax.set_ylim(-0.05, 1.15)
    ax.text(0.95, 0.95, f'$t$ = {elapsed_min:.0f} min', color=color, fontsize=9,
            transform=ax.transAxes, ha='right', va='top')


# ── Main figure function ──────────────────────────────────────────────────────

def make_cpmg_figure(sample, temp_str, stem=None):
    """Generate and save 4-panel CPMG figure for one dataset.

    Left column: raw echo decays + fit for the first and last good scan.
    Right column: T₂ and β vs elapsed time (with highlighted markers for the
    two scans shown on the left).  ConnectionPatch arrows link left←right.
    """
    if stem is None:
        stem = f'SI_figures/cpmg_fit_{sample}_{temp_str}'

    df       = pd.read_csv(pu.NMR_RAW)
    all_mask = (df['sample'] == sample) & (df['temp'] == temp_str)
    sub      = df[all_mask].dropna(subset=['T2', 'beta']).sort_values('elapsed_min')

    good    = sub[~sub['dropped']] if 'dropped' in sub.columns else sub
    dropped = sub[sub['dropped']]  if 'dropped' in sub.columns else sub.iloc[0:0]

    t    = good['elapsed_min'].to_numpy()
    T2   = good['T2'].to_numpy()
    T2e  = _cap_err(good['T2'], good['T2_err'])
    beta = good['beta'].to_numpy()
    be   = _cap_err(good['beta'], good['beta_err'])

    first_row = good.iloc[0]
    last_row  = good.iloc[-1]

    # ── Load raw decay data ───────────────────────────────────────────────────
    decay_available = True
    try:
        t_f, y_f = _load_scan_decay(sample, temp_str,
                                    int(first_row['scan']), int(first_row['n_avg']))
        t_l, y_l = _load_scan_decay(sample, temp_str,
                                    int(last_row['scan']),  int(last_row['n_avg']))
    except (FileNotFoundError, KeyError, zipfile.BadZipFile, Exception) as exc:
        print(f'[WARN] Could not load raw decay data: {exc}')
        decay_available = False

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 5.0))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 1.8],
                           wspace=0.28, hspace=0.12)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_tr)

    # ── Right panels: T₂ and β vs time ───────────────────────────────────────
    ax_tr.errorbar(t, T2, yerr=T2e,
                   fmt='o', ms=4, color='k', mfc='k',
                   capsize=3, lw=0.8, elinewidth=0.8)
    ax_tr.set_yscale('log')
    ax_tr.set_ylim(*T2_LIM)
    ax_tr.set_ylabel(r'$T_2$ (s)')

    ax_br.errorbar(t, beta, yerr=be,
                   fmt='o', ms=4, color='k', mfc='k',
                   capsize=3, lw=0.8, elinewidth=0.8)
    ax_br.set_ylim(*BETA_LIM)
    ax_br.set_ylabel(r'$\beta$')
    ax_br.set_xlabel('Elapsed cure time (min)')

    # Dropped scans
    if not dropped.empty:
        ax_tr.scatter(dropped['elapsed_min'], dropped['T2'].clip(*T2_LIM),
                      color='red', alpha=0.4, zorder=3, s=20)
        ax_br.scatter(dropped['elapsed_min'], dropped['beta'].clip(*BETA_LIM),
                      color='red', alpha=0.4, zorder=3, s=20)

    _asterisks(ax_tr, t, T2,   *T2_LIM)
    _asterisks(ax_br, t, beta, *BETA_LIM)

    # Co-adding shading
    if 'n_avg' in sub.columns and (sub['n_avg'] > 1).any():
        avg_rows = sub[sub['n_avg'] > 1]
        t_shade  = avg_rows['elapsed_min'].min()
        t_end    = sub['elapsed_min'].max()
        for ax in (ax_tr, ax_br):
            ax.axvspan(t_shade, t_end, alpha=0.08, color='gray', zorder=0)

    # Highlighted stars offset above the corresponding data points
    T2_span   = np.log10(T2_LIM[1]) - np.log10(T2_LIM[0])   # log-scale offset
    T2_offset = 10 ** (np.log10(first_row['T2']) + 0.10 * T2_span)
    T2_offset_last = 10 ** (np.log10(last_row['T2']) + 0.10 * T2_span)
    beta_offset = 0.10 * (BETA_LIM[1] - BETA_LIM[0])

    ax_tr.scatter(first_row['elapsed_min'], T2_offset,
                  s=55, marker='*', color=COLOR_FIRST, zorder=7, linewidths=0)
    ax_tr.scatter(last_row['elapsed_min'],  T2_offset_last,
                  s=55, marker='*', color=COLOR_LAST,  zorder=7, linewidths=0)
    ax_br.scatter(first_row['elapsed_min'], first_row['beta'] + beta_offset,
                  s=55, marker='*', color=COLOR_FIRST, zorder=7, linewidths=0)
    ax_br.scatter(last_row['elapsed_min'],  last_row['beta']  + beta_offset,
                  s=55, marker='*', color=COLOR_LAST,  zorder=7, linewidths=0)

    # ── Left panels: echo decays ──────────────────────────────────────────────
    if decay_available:
        _plot_decay_panel(ax_tl, t_f, y_f,
                          float(first_row['A']), float(first_row['T2']),
                          float(first_row['beta']), COLOR_FIRST,
                          float(first_row['elapsed_min']))
        _plot_decay_panel(ax_bl, t_l, y_l,
                          float(last_row['A']),  float(last_row['T2']),
                          float(last_row['beta']),  COLOR_LAST,
                          float(last_row['elapsed_min']))

    else:
        for ax, color, row in [(ax_tl, COLOR_FIRST, first_row),
                               (ax_bl, COLOR_LAST,  last_row)]:
            ax.text(0.5, 0.5, f'Raw data\nnot available\n(t = {row["elapsed_min"]:.0f} min)',
                    transform=ax.transAxes, ha='center', va='center',
                    color=color, fontsize=8)
            ax.set_xlabel('Echo center time (ms)')
            ax.set_ylabel('Normalised intensity')

    # Panel labels: a top-left, b bottom-left, c top-right, d bottom-right
    for ax, label in [(ax_tl, '(a)'), (ax_bl, '(b)'),
                      (ax_tr, '(c)'), (ax_br, '(d)')]:
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                va='top', ha='left', fontweight='bold', fontsize=9)

    plt.setp(ax_tr.get_xticklabels(), visible=False)

    pu.savefig(fig, stem)
    plt.close(fig)
    return stem


def main():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    })
    pu.snapshot_data()
    make_cpmg_figure(SAMPLE, TEMP_STR, stem='cpmg_fit')
    pu.write_provenance(
        __file__,
        datasets=[f'NMR raw CPMG fit results — sample: {SAMPLE}, temp: {TEMP_STR} (T₂ and β vs elapsed time)'],
        source_paths=[pu.NMR_RAW],
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=None,
                        help='Redirect all output here instead of figures/ (does not touch the originals).')
    args = parser.parse_args()
    if args.outdir:
        pu.set_output_dir(args.outdir)
    main()
