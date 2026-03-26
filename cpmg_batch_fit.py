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
            intdata = np.abs(timedata).sum(axis=0)
        else:
            intdata = timedata.sum(axis=0)
        nr_echoes = y_dim
    else:
        if np.iscomplexobj(data):
            intdata = np.abs(data)
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

def stretched_exp(t, A, T2, beta, c):
    return A * np.exp(-(t / T2) ** beta) + c


def fit_scan(scan: dict, zf: zipfile.ZipFile,
             prev: dict | None = None,
             A_first: float | None = None) -> dict | None:
    """
    Fit one scan to A * exp(-(t/T2)^beta).

    A is always initialised from the data (max of signal).
    T2 and beta are warm-started from prev if available.
    All scans are returned; dropped=True marks poor fits (T2 rel. unc. > 200%).
    Dropped scans are excluded from warm-starting the next fit.
    """
    idx = scan["index"]
    try:
        par_bytes  = zf.read(scan["par_info"].filename)
        data_bytes = zf.read(scan["data_info"].filename)

        params = _read_params_from_bytes(par_bytes)
        t, y   = _prepare_t2_from_bytes(data_bytes, params)
        ts     = datetime(*scan["data_info"].date_time)

        A0 = float(np.max(y)) if np.max(y) > 0 else 1.0
        # c0: estimate noise floor from last 10% of echoes
        c0 = float(np.mean(y[int(0.9 * len(y)):]))

        bounds = ([0, 1e-6, 0.1, 0], [np.inf, np.inf, np.inf, np.inf])

        # A always from data; T2/beta/c warm-started from prev if available
        T2_0   = prev["T2"]   if prev is not None else t[len(t) // 2]
        beta_0 = prev["beta"] if prev is not None else 0.8
        c_0    = prev["c"]    if prev is not None else max(c0, 0.0)
        p0     = [A0, T2_0, beta_0, c_0]

        try:
            popt, pcov = curve_fit(stretched_exp, t, y, p0=p0,
                                   bounds=bounds, maxfev=5000)
        except RuntimeError:
            # Warm start failed — retry with data-derived guesses
            popt, pcov = curve_fit(stretched_exp, t, y,
                                   p0=[A0, t[len(t) // 2], 0.8, max(c0, 0.0)],
                                   bounds=bounds, maxfev=5000)

        perr = np.sqrt(np.diag(pcov))
        dropped = bool(popt[1] > 0 and perr[1] / popt[1] > 2.0)
        if dropped:
            print(f"  [DROP] scan {idx}: T2 rel. unc. = {perr[1]/popt[1]:.2f}")

        return {
            "scan":      idx,
            "timestamp": ts,
            "A":         popt[0], "A_err":    perr[0],
            "T2":        popt[1], "T2_err":   perr[1],
            "beta":      popt[2], "beta_err": perr[2],
            "c":         popt[3], "c_err":    perr[3],
            "dropped":   dropped,
            "_t": t, "_y": y,
        }
    except Exception as e:
        print(f"  [WARN] scan {idx}: {e}")
        return None


# ── Per-sample batch runner ───────────────────────────────────────────────────

def process_sample(zip_path: Path, wm_id: str, temp: str, sample: str,
                   direct_prefix: str | None = None) -> pd.DataFrame:
    """Load all scans for one sample from a zip and fit them sequentially,
    warm-starting each fit from the previous successful result.
    If direct_prefix is given, uses list_scans_direct instead of list_scans."""
    print(f"Processing {sample} ({wm_id}) at {temp}...")
    with zipfile.ZipFile(zip_path) as zf:
        if direct_prefix is not None:
            scans = list_scans_direct(zf, direct_prefix)
        else:
            scans = list_scans(zf, wm_id)
        print(f"  Found {len(scans)} scans.")

        results = []
        prev    = None
        A_first = None
        for scan in scans:
            result = fit_scan(scan, zf, prev=prev, A_first=A_first)
            if result is not None:
                if A_first is None:
                    A_first = result["A"]
                results.append(result)
                if not result["dropped"]:
                    prev = result

    results = [r for r in results if r is not None]
    for r in results:
        r.pop("_t", None)
        r.pop("_y", None)
    df = pd.DataFrame(results)
    # keep dropped column for plotting; strip before CSV save
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
    Add alpha = 1 - T2 / T2_initial to each sample group.
    Uses the first non-dropped scan's T2 as T2_initial.
    Only operates on good (non-dropped) fits.
    """
    df = df.copy()
    df["alpha"] = np.nan
    for (sample, temp), grp in df.groupby(["sample", "temp"]):
        good = grp[~grp["dropped"]] if "dropped" in grp.columns else grp
        if good.empty:
            continue
        T2_0 = good["T2"].iloc[0]
        alpha = 1.0 - good["T2"] / T2_0
        df.loc[good.index, "alpha"] = alpha.values
    return df


def t2_alpha_model(alpha, B, a0):
    """T2(alpha) = T2_0 * exp(B * alpha / (a0 - alpha))"""
    return np.exp(B * alpha / (a0 - alpha))   # normalised: divide by T2_0


def fit_t2_alpha(df: pd.DataFrame, sample: str, temp: str) -> dict | None:
    """
    Fit T2/T2_0 vs alpha to the model exp(B*alpha/(a0-alpha)).
    Returns dict with B, a0 and their uncertainties, or None if fit fails.
    """
    grp = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if "dropped" in grp.columns:
        grp = grp[~grp["dropped"]]
    grp = grp.dropna(subset=["alpha"])
    if len(grp) < 3:
        return None

    T2_0  = grp["T2"].iloc[0]
    alpha = grp["alpha"].values
    y     = grp["T2"].values / T2_0   # normalised

    # Exclude alpha >= a0 to avoid singularity; a0 must be > max(alpha)
    alpha_max = alpha.max()

    try:
        popt, pcov = curve_fit(
            t2_alpha_model, alpha, y,
            p0=[-2.0, min(alpha_max * 1.1, 0.99)],
            bounds=([-np.inf, alpha_max + 1e-6], [0, np.inf]),
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
        print(f"  [WARN] T2(alpha) fit failed for {sample} {temp}: {e}")
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

    ax1.errorbar(good["elapsed_min"], good["T2"],
                 yerr=_cap_err(good["T2"], good["T2_err"]),
                 fmt="o-", capsize=3)
    ax1.scatter(dropped["elapsed_min"], dropped["T2"],
                color="red", alpha=0.4, zorder=3, s=20)
    ax1.set_ylabel("T₂ (s)")

    ax2.errorbar(good["elapsed_min"], good["beta"],
                 yerr=_cap_err(good["beta"], good["beta_err"]),
                 fmt="o-", capsize=3, color="C1")
    ax2.scatter(dropped["elapsed_min"], dropped["beta"],
                color="red", alpha=0.4, zorder=3, s=20)
    ax2.set_ylabel("β")
    ax2.set_xlabel("Elapsed time (min)")

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
        prev    = None
        A_first = None
        for scan in scans:
            r = fit_scan(scan, zf, prev=prev, A_first=A_first)
            if r is not None:
                if A_first is None:
                    A_first = r["A"]
                results.append(r)
                prev = r

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
        y_fit = stretched_exp(t, r["A"], r["T2"], r["beta"], r["c"])

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


def plot_t2_alpha(df: pd.DataFrame, fit: dict, out_dir: Path) -> None:
    """Plot T2/T2_0 vs alpha with the fitted model curve."""
    sample, temp = fit["sample"], fit["temp"]
    grp = df[(df["sample"] == sample) & (df["temp"] == temp)]
    if "dropped" in grp.columns:
        grp = grp[~grp["dropped"]]
    grp = grp.dropna(subset=["alpha"])
    if grp.empty:
        return

    T2_0  = fit["T2_0"]
    alpha = grp["alpha"].values
    y     = grp["T2"].values / T2_0

    alpha_fine = np.linspace(0, alpha.max(), 300)
    # avoid singularity at a0
    alpha_fine = alpha_fine[alpha_fine < fit["a0"] - 1e-6]
    y_model    = t2_alpha_model(alpha_fine, fit["B"], fit["a0"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(alpha, y, s=20, zorder=3, label="data")
    ax.plot(alpha_fine, y_model, "-", color="C1",
            label=f"B={fit['B']:.3f}, a₀={fit['a0']:.3f}")
    ax.set_xlabel("α = 1 − T₂/T₂₀")
    ax.set_ylabel("T₂ / T₂₀")
    ax.set_title(f"{sample} — {temp}")
    ax.legend()
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

            for (temp, sample), grp in csv_df.groupby(["temp", "sample"]):
                grp.to_csv(out_dir / f"{sample}_{temp}.csv", index=False)
                plot_sample(csv_df, sample, temp, out_dir)

            # Fit T2(alpha) model for each sample
            t2a_results = []
            for (temp, sample) in csv_df.groupby(["temp", "sample"]).groups:
                result = fit_t2_alpha(csv_df, sample, temp)
                if result:
                    t2a_results.append(result)
                    plot_t2_alpha(csv_df, result, out_dir)

            if t2a_results:
                t2a_df = pd.DataFrame(t2a_results)
                t2a_df.to_csv(out_dir / "t2_alpha_fits.csv", index=False)
                print("\nT2(alpha) fit results:")
                print(t2a_df.to_string(index=False))

            good_df = csv_df[~csv_df["dropped"]] if "dropped" in csv_df.columns else csv_df
            print(good_df.groupby(["temp", "sample"])[["T2", "beta"]].describe().round(4))
