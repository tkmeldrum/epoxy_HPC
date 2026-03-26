"""
cpmg_batch_fit.py
-----------------
Load CPMG data from zipped Kea datasets, fit each scan to a stretched
exponential, and save results with timestamps.

Stretched exponential: M(t) = A * exp(-(t/T2)^beta)

Results are saved per-sample to CSV files in cpmg_fit_results/.
"""

import sys
import zipfile
import io
import struct
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from multiprocessing import Pool

# ── Machine-specific paths ────────────────────────────────────────────────────
import local_config
sys.path.insert(0, local_config.KEA_PATH)
from kea_io import prepare_t2_data

# ── Configuration ─────────────────────────────────────────────────────────────
ZIP_ROOT = Path(local_config.DATA_ROOT)

# temp -> {sample_name -> WM number (zero-padded to 2 digits)}
EXPERIMENTS = {
    "25C": {"EDA": "WM39", "DAP": "WM38", "DAB": "WM40"},
    "33C": {"EDA": "WM66", "DAP": "WM63", "DAB": "WM65"},
    "40C": {"EDA": "WM58", "DAP": "WM54", "DAB": "WM53"},
}

N_WORKERS = local_config.N_WORKERS

# DAP2: separate zip with a different internal folder structure
DAP2_ZIP = ZIP_ROOT / "DAP2.zip"
DAP2_EXPERIMENTS = {
    "25C": "Epoxy2026/13DAP/CPMG_25C",
    "33C": "Epoxy2026/13DAP/CPMG_33C",
    "40C": "Epoxy2026/13DAP/CPMG_40C_2",
}


# ── Zip-aware Kea readers ─────────────────────────────────────────────────────

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
    """Parse acqu.par content from raw bytes."""
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
    """Rotate complex array so real sum is maximised and imag sum ≈ 0."""
    if not np.iscomplexobj(d):
        return d
    theta = -np.angle(d.sum())
    return d * np.exp(1j * theta)


def _prepare_t2_from_bytes(data_bytes: bytes, params: dict):
    """
    Wrap prepare_t2_data to work on in-memory bytes rather than a file path.
    Returns (echo_vec, intdata) just like kea_io.prepare_t2_data.
    """
    # prepare_t2_data calls read_kea_binary(filepath) internally.
    # We replicate the logic here on a BytesIO object.
    f = io.BytesIO(data_bytes)

    f.read(4)   # owner
    f.read(4)   # format
    f.read(4)   # version
    data_type = struct.unpack("<i", f.read(4))[0]
    x_dim     = struct.unpack("<i", f.read(4))[0]
    y_dim     = struct.unpack("<i", f.read(4))[0]
    f.read(4)   # z_dim
    f.read(4)   # q_dim

    total = x_dim * y_dim  # z/q are 1 for standard CPMG

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
        echo_vec_cpx = timedata.sum(axis=0)          # coherent sum → one complex value per echo
        intdata = np.real(_autophase(echo_vec_cpx))
        nr_echoes = y_dim
    else:
        intdata = np.real(_autophase(data)) if np.iscomplexobj(data) else np.asarray(data, dtype=float)
        nr_echoes = x_dim

    echo_vec = 1e-6 * echo_time * np.arange(1, nr_echoes + 1)
    return echo_vec, intdata


# ── Scan discovery ────────────────────────────────────────────────────────────

def list_scans(zf: zipfile.ZipFile, wm_id: str) -> list[dict]:
    """
    Find all numbered scan folders for a given WMXX inside an open ZipFile.
    Returns list of dicts sorted by scan index.
    """
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

    # Keep only complete scans
    complete = [s for s in scans.values() if s["data_info"] and s["par_info"]]
    complete.sort(key=lambda s: s["index"])
    return complete


def list_scans_direct(zf: zipfile.ZipFile, prefix: str) -> list[dict]:
    """
    Find numbered scan folders under an arbitrary prefix inside a ZipFile.
    Used for DAP2 whose structure is Epoxy2026/13DAP/CPMG_XXC/<n>/data.2d
    """
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


# ── Stretched exponential fit ─────────────────────────────────────────────────

# include y-offset
# def stretched_exp(t, A, T2, beta, c):
#     return A * np.exp(-(t / T2) ** beta) + c

# omit y-offset
def stretched_exp(t, A, T2, beta):
    return A * np.exp(-(t / T2) ** beta)


def _read_scan_data(scan: dict, zf: zipfile.ZipFile):
    """Read and autophase one scan. Returns (t, y, ts, idx) or None."""
    idx = scan["index"]
    try:
        par_bytes  = zf.read(scan["par_info"].filename)
        data_bytes = zf.read(scan["data_info"].filename)
        params = _read_params_from_bytes(par_bytes)
        t, y   = _prepare_t2_from_bytes(data_bytes, params)
        ts     = datetime(*scan["data_info"].date_time)
        return t, y, ts, idx
    except Exception as e:
        print(f"  [WARN] scan {idx} (read): {e}")
        return None


def _fit_echo_array(t, y, ts, idx, n_avg=1) -> dict | None:
    """Fit a (possibly averaged) echo array to A * exp(-(t/T2)^beta)."""
    try:
        A0 = float(np.max(y)) if np.max(y) > 0 else 1.0
        half_max_idx = np.searchsorted(-y, -A0 / 2)
        T2_0 = float(t[min(half_max_idx, len(t) - 1)])
        p0     = [A0, T2_0, 1.0]
        bounds = ([0, 0, 0], [np.inf, np.inf, 5])
        popt, pcov = curve_fit(stretched_exp, t, y, p0=p0,
                               bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        drop_unc  = popt[1] > 0 and perr[1] / popt[1] > 2.0
        drop_beta = popt[2] > 2.0
        dropped = bool(drop_unc or drop_beta)
        if drop_unc:
            print(f"  [DROP] scan {idx}: T2 rel. unc. = {perr[1]/popt[1]:.2f}")
        if drop_beta:
            print(f"  [DROP] scan {idx}: beta = {popt[2]:.2f} > 2")
        return {
            "scan":      idx,
            "timestamp": ts,
            "n_avg":     n_avg,
            "A":         popt[0], "A_err":    perr[0],
            "T2":        popt[1], "T2_err":   perr[1],
            "beta":      popt[2], "beta_err": perr[2],
            "dropped":   dropped,
            "_t": t, "_y": y,
        }
    except Exception as e:
        print(f"  [WARN] scan {idx} (fit): {e}")
        return None


def fit_scan(scan: dict, zf: zipfile.ZipFile) -> dict | None:
    """Read and fit one scan. Thin wrapper used by plot_decays."""
    data = _read_scan_data(scan, zf)
    if data is None:
        return None
    return _fit_echo_array(*data)


# ── Per-sample batch runner ───────────────────────────────────────────────────

N_AVG_TARGET  = 20    # target number of averaged points in the low-T2 tail
T2_AVG_THRESH = 0.2  # fraction of T2(0) below which averaging kicks in


def process_sample(zip_path: Path, wm_id: str, temp: str, sample: str,
                   direct_prefix: str | None = None) -> pd.DataFrame:
    """Load all scans for one sample and fit with adaptive tail averaging.

    Pass 1: fit every scan individually.
    Find the first scan where T2 < T2_AVG_THRESH * T2(0) (first reliable T2);
    everything from that point onward is re-binned into ~N_AVG_TARGET averaged points.
    """
    print(f"Processing {sample} ({wm_id}) at {temp}...")
    with zipfile.ZipFile(zip_path) as zf:
        if direct_prefix is not None:
            scans = list_scans_direct(zf, direct_prefix)
        else:
            scans = list_scans(zf, wm_id)
        print(f"  Found {len(scans)} scans.")
        scan_data = [d for scan in scans
                     if (d := _read_scan_data(scan, zf)) is not None]

    if not scan_data:
        return pd.DataFrame()

    # Pass 1: fit all individually
    results_p1 = [_fit_echo_array(*d) for d in scan_data]

    # T2_0: first non-dropped scan with T2_err/T2 < 0.5
    T2_0_result = next(
        (r for r in results_p1
         if r is not None and not r["dropped"] and r["T2"] > 0
         and r["T2_err"] / r["T2"] < 0.5),
        None
    )

    if T2_0_result is None:
        results = [r for r in results_p1 if r is not None]
    else:
        T2_0      = T2_0_result["T2"]
        threshold = T2_AVG_THRESH * T2_0
        print(f"  T2(0)={T2_0:.4f} s  →  threshold={threshold:.4f} s ({T2_AVG_THRESH*100:.0f}%)")

        # First position where T2 drops below threshold — everything after is averaged
        switch_pos = next(
            (i for i, r in enumerate(results_p1) if r is not None and r["T2"] < threshold),
            None
        )
        if switch_pos is not None:
            sr = results_p1[switch_pos]
            print(f"  Switch at scan {sr['scan']}, T2={sr['T2']:.4f} s"
                  f"  ({len(results_p1) - switch_pos} scans → averaged tail)")

        if switch_pos is None:
            results = [r for r in results_p1 if r is not None]
        else:
            high_results = [r for r in results_p1[:switch_pos] if r is not None]
            low_data     = scan_data[switch_pos:]
            n_low        = len(low_data)
            n_avg        = max(1, round(n_low / N_AVG_TARGET))
            n_bins       = n_low // n_avg
            print(f"  Low-T2 tail: {n_low} scans → bins of {n_avg} (~{n_bins} points)")

            # Split low_data into contiguous runs of equal echo count
            echo_runs = []
            for d in low_data:
                if echo_runs and len(d[1]) == len(echo_runs[-1][0][1]):
                    echo_runs[-1].append(d)
                else:
                    echo_runs.append([d])

            # Within each run, build groups of n_avg; fold remainder into last
            groups = []
            for run in echo_runs:
                n_run  = len(run)
                n_full = n_run // n_avg
                g = [run[i * n_avg:(i + 1) * n_avg] for i in range(n_full)]
                remainder = run[n_full * n_avg:]
                if remainder:
                    if g:
                        g[-1] = g[-1] + remainder
                    else:
                        g = [remainder]
                groups.extend(g)

            low_results = []
            for group in groups:
                t     = group[0][0]
                y_avg = np.mean([d[1] for d in group], axis=0)
                ts    = group[len(group) // 2][2]   # midpoint timestamp
                idx   = group[0][3]
                r = _fit_echo_array(t, y_avg, ts, idx, n_avg=len(group))
                if r is not None:
                    low_results.append(r)

            results = high_results + low_results

    for r in results:
        r.pop("_t", None)
        r.pop("_y", None)

    df = pd.DataFrame(results)
    df.insert(0, "sample", sample)
    df.insert(1, "temp",   temp)

    if not df.empty:
        t0 = df["timestamp"].iloc[0]
        df["elapsed_min"] = [(ts - t0).total_seconds() / 60 for ts in df["timestamp"]]

    return df


# ── Note on multiprocessing ───────────────────────────────────────────────────
# curve_fit is CPU-bound, but ZipFile objects can't be pickled across processes.
# Options:
#   A) Parallelize at the sample level (9 samples → 6 workers): easy, good speedup.
#   B) Extract bytes first, then scatter to workers: more complex but parallelizes
#      within a sample too.
# For now we use option A via a simple loop (samples are already fast individually).
# Uncomment the Pool block below to enable sample-level parallelism if needed.

def run_all() -> pd.DataFrame:
    all_dfs = []

    # Standard samples (EDA, DAP, DAB)
    for temp, samples in EXPERIMENTS.items():
        zip_path = ZIP_ROOT / f"{temp}.zip"
        if not zip_path.exists():
            print(f"[SKIP] {zip_path} not found.")
            continue
        for sample, wm_id in samples.items():
            df = process_sample(zip_path, wm_id, temp, sample)
            all_dfs.append(df)

    # DAP2
    if not DAP2_ZIP.exists():
        print(f"[SKIP] {DAP2_ZIP} not found.")
    else:
        for temp, prefix in DAP2_EXPERIMENTS.items():
            df = process_sample(DAP2_ZIP, "DAP2", temp, "DAP2",
                                direct_prefix=prefix)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── Alpha and T2(alpha) model ─────────────────────────────────────────────────

def compute_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add alpha = 1 - T2 / max(T2) to each sample group.
    Alpha is only assigned to scans at or after the max-T2 point (post-peak),
    so pre-peak scans (T2 still rising) are excluded from downstream analysis.
    Only operates on good (non-dropped) fits.
    """
    df = df.copy()
    df["alpha"] = np.nan
    for (sample, temp), grp in df.groupby(["sample", "temp"]):
        good = grp[~grp["dropped"]] if "dropped" in grp.columns else grp
        if good.empty:
            continue
        reliable   = good[good["T2_err"] / good["T2"] < 0.5]
        ref        = reliable if not reliable.empty else good
        T2_max     = ref["T2"].max()
        peak_idx   = ref["T2"].idxmax()
        post_peak  = good.loc[peak_idx:]
        alpha      = 1.0 - post_peak["T2"] / T2_max
        df.loc[post_peak.index, "alpha"] = alpha.values
    return df


def r2_alpha_model(alpha, B, a0):
    """R2(alpha) = R2_0 * exp(B * alpha / (a0 - alpha));  B > 0"""
    return np.exp(B * alpha / (a0 - alpha))   # normalised: R2 / R2_0


def fit_t2_alpha(df: pd.DataFrame, sample: str, temp: str) -> dict | None:
    """
    Fit R2/R2_0 vs alpha to R2_alpha_model (B > 0).
    R2_0 = 1/max(T2) (most liquid state, alpha=0 reference).
    Returns dict with B, a0 and their uncertainties, or None if fit fails.
    """
    grp = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if "dropped" in grp.columns:
        grp = grp[~grp["dropped"]]
    grp = grp.dropna(subset=["alpha"])
    if len(grp) < 3:
        return None

    T2_0  = grp["T2"].max()   # max(T2) → R2_0 = 1/T2_0
    alpha = grp["alpha"].values
    y     = T2_0 / grp["T2"].values   # R2 / R2_0 = T2_0 / T2

    alpha_max = alpha.max()

    try:
        popt, pcov = curve_fit(
            r2_alpha_model, alpha, y,
            p0=[2.0, alpha_max * 1.05],
            bounds=([0, alpha_max + 1e-6], [np.inf, np.inf]),
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "sample": sample, "temp": temp,
            "T2_0":   T2_0,
            "B":      popt[0], "B_err":  perr[0],
            "a0":     popt[1], "a0_err": perr[1],
        }
    except Exception as e:
        print(f"  [WARN] R2(alpha) fit failed for {sample} {temp}: {e}")
        return None


# ── Plotting ─────────────────────────────────────────────────────────────────

def _cap_err(vals: pd.Series, errs: pd.Series, max_relative: float = 0.2) -> pd.Series:
    """Cap error bars at max_relative * |value| to keep plots legible."""
    return np.minimum(errs, max_relative * vals.abs())


def plot_sample(df: pd.DataFrame, sample: str, temp: str, out_dir: Path) -> None:
    """T2 and beta vs elapsed time for one sample, saved as PNG.
    Good fits shown as solid lines; dropped scans shown as faded red markers."""
    grp = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if grp.empty:
        return

    good    = grp[~grp["dropped"]] if "dropped" in grp.columns else grp
    dropped = grp[grp["dropped"]]  if "dropped" in grp.columns else grp.iloc[0:0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    fig.suptitle(f"{sample} — {temp}")

    T2_YLIM   = (0, 0.05)
    BETA_YLIM = (0, 2.1)

    ax1.errorbar(good["elapsed_min"], good["T2"],
                 yerr=_cap_err(good["T2"], good["T2_err"]),
                 fmt="o-", capsize=3)
    ax1.scatter(dropped["elapsed_min"], dropped["T2"],
                color="red", alpha=0.4, zorder=3, s=20)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, T2_YLIM[1])
    ax1.set_ylabel("T₂ (s)")
    # asterisks for out-of-range T2 points
    clipped_t2 = grp[grp["T2"] > T2_YLIM[1]]
    if not clipped_t2.empty:
        ax1.scatter(clipped_t2["elapsed_min"],
                    [T2_YLIM[1] * 0.97] * len(clipped_t2),
                    marker="*", color="C0", s=60, zorder=5, clip_on=False)

    ax2.errorbar(good["elapsed_min"], good["beta"],
                 yerr=_cap_err(good["beta"], good["beta_err"]),
                 fmt="o-", capsize=3, color="C1")
    ax2.scatter(dropped["elapsed_min"], dropped["beta"],
                color="red", alpha=0.4, zorder=3, s=20)
    ax2.set_ylim(*BETA_YLIM)
    ax2.set_ylabel("β")
    ax2.set_xlabel("Elapsed time (min)")
    # asterisks for out-of-range beta points
    clipped_beta = grp[grp["beta"] > BETA_YLIM[1]]
    if not clipped_beta.empty:
        ax2.scatter(clipped_beta["elapsed_min"],
                    [BETA_YLIM[1] * 0.97] * len(clipped_beta),
                    marker="*", color="C1", s=60, zorder=5, clip_on=False)

    # Shade averaged tail region
    if "n_avg" in grp.columns and (grp["n_avg"] > 1).any():
        avg_rows  = grp[grp["n_avg"] > 1]
        t_shade   = avg_rows["elapsed_min"].min()
        t_end     = grp["elapsed_min"].max()
        n_label   = int(avg_rows["n_avg"].median())
        for ax in (ax1, ax2):
            ax.axvspan(t_shade, t_end, alpha=0.08, color="gray", zorder=0)
        ax1.text(t_shade, T2_YLIM[1] * 0.95, f"N≈{n_label} avg",
                 fontsize=7, va="top", ha="left", color="gray")

    fig.tight_layout()
    fig.savefig(out_dir / f"{sample}_{temp}.png", dpi=150)
    plt.close(fig)


# ── Decay diagnostic ─────────────────────────────────────────────────────────

def plot_decays(zip_path: Path, wm_id: str, temp: str, sample: str,
                out_dir: Path, direct_prefix: str | None = None) -> None:
    """
    For each scan in a sample, plot the raw CPMG decay and the stretched
    exponential fit on a shared grid. One PNG per sample saved to out_dir/decays/.
    Dropped scans are highlighted in red.
    """
    print(f"Decay diagnostics: {sample} ({wm_id}) at {temp}...")
    with zipfile.ZipFile(zip_path) as zf:
        if direct_prefix is not None:
            scans = list_scans_direct(zf, direct_prefix)
        else:
            scans = list_scans(zf, wm_id)
        results = []
        for scan in scans:
            r = fit_scan(scan, zf)
            if r is not None:
                results.append(r)

    if not results:
        print("  No results.")
        return

    n = len(results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        t, y = r["_t"], r["_y"]
        # y_fit = stretched_exp(t, r["A"], r["T2"], r["beta"], r["c"])
        y_fit = stretched_exp(t, r["A"], r["T2"], r["beta"])

        color = "red" if r.get("dropped") else "C0"
        ax.plot(t * 1000, y, ".", ms=3, color=color, alpha=0.6)
        ax.plot(t * 1000, y_fit, "-", color="k", lw=1)
        ax.set_title(
            f"#{r['scan']}  T₂={r['T2']*1000:.2f}ms  β={r['beta']:.2f}",
            fontsize=7, color=color
        )
        ax.tick_params(labelsize=6)
        ax.set_xlabel("t (ms)", fontsize=6)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{sample} — {temp}  |  red = dropped", fontsize=10)
    fig.tight_layout()

    decay_dir = out_dir / "decays"
    decay_dir.mkdir(exist_ok=True)
    fig.savefig(decay_dir / f"{sample}_{temp}_decays.png", dpi=120)
    plt.close(fig)
    print(f"  Saved to {decay_dir / f'{sample}_{temp}_decays.png'}")


def plot_t2_alpha(df: pd.DataFrame, sample: str, temp: str,
                  out_dir: Path, fit: dict | None = None) -> None:
    """Alpha vs time (always) and T2/T2_max vs alpha with fit (when fit succeeds)."""
    grp_all = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if "dropped" in grp_all.columns:
        grp_all = grp_all[~grp_all["dropped"]]

    grp = grp_all.dropna(subset=["alpha"])
    if grp.empty:
        return

    alpha = grp["alpha"].values
    has_fit = fit is not None

    fig, axes = plt.subplots(1, 2 if has_fit else 1,
                             figsize=(10, 4) if has_fit else (5, 4))
    ax1 = axes[0] if has_fit else axes
    fig.suptitle(f"{sample} — {temp}")

    # Left: alpha vs elapsed time
    ax1.scatter(grp["elapsed_min"], alpha, s=20, zorder=3)
    ax1.set_xlabel("Elapsed time (min)")
    ax1.set_ylabel("α = 1 − T₂ / max(T₂)")
    ax1.set_ylim(0, None)

    # Right: R2/R2_0 vs alpha with model (only when fit succeeded)
    if has_fit:
        T2_0       = fit["T2_0"]
        y          = T2_0 / grp["T2"].values   # R2 / R2_0
        alpha_fine = np.linspace(0, alpha.max(), 300)
        alpha_fine = alpha_fine[alpha_fine < fit["a0"] - 1e-6]
        y_model    = r2_alpha_model(alpha_fine, fit["B"], fit["a0"])

        ax2 = axes[1]
        ax2.scatter(alpha, y, s=20, zorder=3, label="data")
        ax2.plot(alpha_fine, y_model, "-", color="C1",
                 label=f"B = {fit['B']:.3f} ± {fit['B_err']:.3f}\n"
                       f"a₀ = {fit['a0']:.3f} ± {fit['a0_err']:.3f}")
        ax2.set_xlabel("α")
        ax2.set_ylabel("R₂ / R₂₀")
        ax2.set_ylim(1, None)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    t2a_dir = out_dir / "t2_alpha"
    t2a_dir.mkdir(exist_ok=True)
    fig.savefig(t2a_dir / f"{sample}_{temp}_t2alpha.png", dpi=150)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", nargs=2, metavar=("SAMPLE", "TEMP"),
                        help="Plot decay diagnostics for one sample, e.g. --diagnose EDA 40C")
    args = parser.parse_args()

    out_dir = Path("cpmg_fit_results")
    out_dir.mkdir(exist_ok=True)

    if args.diagnose:
        sample, temp = args.diagnose
        if sample == "DAP2":
            if temp not in DAP2_EXPERIMENTS:
                print(f"Unknown temp for DAP2: {temp}. Valid: {list(DAP2_EXPERIMENTS)}.")
            else:
                plot_decays(DAP2_ZIP, "DAP2", temp, "DAP2", out_dir,
                            direct_prefix=DAP2_EXPERIMENTS[temp])
        elif temp not in EXPERIMENTS or sample not in EXPERIMENTS[temp]:
            print(f"Unknown sample/temp: {sample} {temp}. "
                  f"Valid temps: {list(EXPERIMENTS)}, or use DAP2.")
        else:
            zip_path = ZIP_ROOT / f"{temp}.zip"
            wm_id = EXPERIMENTS[temp][sample]
            plot_decays(zip_path, wm_id, temp, sample, out_dir)
    else:
        df = run_all()

        if df.empty:
            print("No results — check zip paths and WM IDs.")
        else:
            # Strip internal columns before saving; keep dropped for plots
            csv_df = df.drop(columns=["_t", "_y"], errors="ignore")

            # Save all (including dropped) so nothing is silently lost
            out_path = out_dir / "all_samples.csv"
            csv_df.to_csv(out_path, index=False)
            print(f"\nSaved {len(df)} fits to {out_path}")

            csv_df = compute_alpha(csv_df)
            csv_df.to_csv(out_path, index=False)   # overwrite with alpha included

            # Per-sample CSVs and summary plots
            for (temp, sample), grp in csv_df.groupby(["temp", "sample"]):
                grp.to_csv(out_dir / f"{sample}_{temp}.csv", index=False)
                try:
                    plot_sample(csv_df, sample, temp, out_dir)
                except Exception as e:
                    print(f"  [WARN] plot_sample failed for {sample} {temp}: {e}")

            # T2(alpha) fit and alpha plots — always plot, fit when possible
            t2a_results = []
            for (temp, sample) in csv_df.groupby(["temp", "sample"]).groups:
                try:
                    result = fit_t2_alpha(csv_df, sample, temp)
                    if result:
                        t2a_results.append(result)
                except Exception as e:
                    print(f"  [WARN] T2(alpha) fit failed for {sample} {temp}: {e}")
                    result = None
                try:
                    plot_t2_alpha(csv_df, sample, temp, out_dir, fit=result)
                except Exception as e:
                    print(f"  [WARN] alpha plot failed for {sample} {temp}: {e}")

            if t2a_results:
                t2a_df = pd.DataFrame(t2a_results)
                t2a_df.to_csv(out_dir / "t2_alpha_fits.csv", index=False)
                print("\nT2(alpha) fit results:")
                print(t2a_df.to_string(index=False))
            else:
                print("\n[WARN] No T2(alpha) fits succeeded.")

            good_df = csv_df[~csv_df["dropped"]] if "dropped" in csv_df.columns else csv_df
            print(good_df.groupby(["temp", "sample"])[["T2", "beta"]].describe().round(4))
