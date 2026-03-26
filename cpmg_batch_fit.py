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

# ── Path to kea_io ────────────────────────────────────────────────────────────
sys.path.insert(0, r"/mnt/c/Users/Tyler Meldrum/Documents/GitHub/MeldrumLabCode/pythonic")
from kea_io import prepare_t2_data

# ── Configuration ─────────────────────────────────────────────────────────────
ZIP_ROOT = Path(
    r"/mnt/c/Users/Tyler Meldrum/OneDrive - William & Mary"
    r"/Documents - Meldrumlab/Epoxy kinetics/Data/Raw data"
)

# temp -> {sample_name -> WM number (zero-padded to 2 digits)}
EXPERIMENTS = {
    "25C": {"EDA": "WM39", "DAP": "WM38", "DAB": "WM40"},
    "33C": {"EDA": "WM66", "DAP": "WM63", "DAB": "WM65"},
    "40C": {"EDA": "WM58", "DAP": "WM54", "DAB": "WM53"},
}

N_WORKERS = 6  # physical cores


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

    def _autophase(d):
        if not np.iscomplexobj(d):
            return d
        theta = -np.angle(d.sum())
        return d * np.exp(1j * theta)

    if y_dim > 1:
        timedata = data[: x_dim * y_dim].reshape((x_dim, y_dim), order="F")
        if np.iscomplexobj(timedata):
            timedata = _autophase(timedata)
            intdata  = np.real(timedata).sum(axis=0)
        else:
            intdata  = timedata.sum(axis=0)
        nr_echoes = y_dim
    else:
        if np.iscomplexobj(data):
            data    = _autophase(data)
            intdata = np.real(data)
        else:
            intdata = np.asarray(data, dtype=float)
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


# ── Stretched exponential fit ─────────────────────────────────────────────────

def stretched_exp(t, A, T2, beta):
    return A * np.exp(-(t / T2) ** beta)


def fit_scan(scan: dict, zf: zipfile.ZipFile) -> dict | None:
    """Fit one scan. Returns result dict or None on failure."""
    idx = scan["index"]
    try:
        par_bytes  = zf.read(scan["par_info"].filename)
        data_bytes = zf.read(scan["data_info"].filename)

        params   = _read_params_from_bytes(par_bytes)
        t, y     = _prepare_t2_from_bytes(data_bytes, params)

        # Timestamp from data.2d ZipInfo
        ts = datetime(*scan["data_info"].date_time)

        A0   = float(y[0]) if y[0] > 0 else float(np.max(y))
        T2_0 = t[len(t) // 2]
        p0     = [A0,    T2_0, 0.8]
        bounds = ([0, 1e-6, 0.1], [np.inf, np.inf, 1.0])

        popt, pcov = curve_fit(stretched_exp, t, y, p0=p0, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))

        return {
            "scan":      idx,
            "timestamp": ts,
            "A":         popt[0], "A_err":    perr[0],
            "T2":        popt[1], "T2_err":   perr[1],
            "beta":      popt[2], "beta_err": perr[2],
        }
    except Exception as e:
        print(f"  [WARN] scan {idx}: {e}")
        return None


# ── Per-sample batch runner ───────────────────────────────────────────────────

def process_sample(zip_path: Path, wm_id: str, temp: str, sample: str) -> pd.DataFrame:
    """Load all scans for one sample from a zip and fit them."""
    print(f"Processing {sample} ({wm_id}) at {temp}...")
    with zipfile.ZipFile(zip_path) as zf:
        scans = list_scans(zf, wm_id)
        print(f"  Found {len(scans)} scans.")
        results = [fit_scan(s, zf) for s in scans]

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.insert(0, "sample", sample)
    df.insert(1, "temp",   temp)

    # Time since first scan (minutes)
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
    for temp, samples in EXPERIMENTS.items():
        zip_path = ZIP_ROOT / f"{temp}.zip"
        if not zip_path.exists():
            print(f"[SKIP] {zip_path} not found.")
            continue
        for sample, wm_id in samples.items():
            df = process_sample(zip_path, wm_id, temp, sample)
            all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_sample(df: pd.DataFrame, sample: str, temp: str, out_dir: Path) -> None:
    """T2 and beta vs elapsed time for one sample, saved as PNG."""
    grp = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if grp.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    fig.suptitle(f"{sample} — {temp}")

    ax1.errorbar(grp["elapsed_min"], grp["T2"], yerr=grp["T2_err"],
                 fmt="o-", capsize=3)
    ax1.set_ylabel("T₂ (s)")

    ax2.errorbar(grp["elapsed_min"], grp["beta"], yerr=grp["beta_err"],
                 fmt="o-", capsize=3, color="C1")
    ax2.set_ylabel("β")
    ax2.set_xlabel("Elapsed time (min)")

    fig.tight_layout()
    fig.savefig(out_dir / f"{sample}_{temp}.png", dpi=150)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path("cpmg_fit_results")
    out_dir.mkdir(exist_ok=True)

    df = run_all()

    if df.empty:
        print("No results — check zip paths and WM IDs.")
    else:
        # Save combined
        out_path = out_dir / "all_samples.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} fits to {out_path}")

        # Save per-sample CSV and plot
        for (temp, sample), grp in df.groupby(["temp", "sample"]):
            grp.to_csv(out_dir / f"{sample}_{temp}.csv", index=False)
            plot_sample(df, sample, temp, out_dir)

        print(df.groupby(["temp", "sample"])[["T2", "beta"]].describe().round(4))
