# epoxy_HPC

## Model notes

The `kapp` model (`BatchBayesian_simple.py`) — which uses `kapp * a^m * (1-a)^(n/2) * (r-a)^(n/2)` — was tested but abandoned. Dropping the k1 term means the reaction rate goes to zero as α→0, which produces poor fits and very wide posterior CIs for datasets that plateau early (e.g. NMR at low temperature). The full Kamal-Malkin form `(k1 + k2*a^m) * (1-a)^(n/2) * (r-a)^(n/2)` is used instead.

## CPMG NMR analysis (`cpmg_batch_fit.py`)

Loads CPMG relaxation data directly from zipped Kea datasets (no extraction needed) and fits each decay to a stretched exponential:

```
M(t) = A * exp(-(t/T2)^beta)
```

**Phasing:** For each scan, all points within each echo are coherently summed to give one complex value per echo. The echo vector is then autophased by rotating by `-angle(sum)` so that the real projection is maximised and the imaginary sum ≈ 0. The real part is used as the decay signal. If the data are real-valued (non-complex), magnitude is used directly.

**Fitting:** Each scan is fit independently using `scipy.optimize.curve_fit` — no warm-starting between scans. Initial guesses: `A = max(y)`, `beta = 1`, `T2` = first time where signal drops to half-max. Bounds: `0 < A, T2 < inf`, `0 < beta < 5`.

**Drop criteria:** A scan is marked `dropped=True` if T2 relative uncertainty > 200% or beta > 2. Dropped scans are excluded from alpha and T2(alpha) analysis but retained in the CSV output. Summary plots clip T2 to [0, 0.05 s] and beta to [0, 2.1]; out-of-range points are marked with an asterisk.

**Machine-specific paths** (zip root, kea_io path, worker count) are set in `local_config.py` — not committed to git. See comments in that file for PC/WSL vs. Mac values.

### Folder/zip structure

Data lives in:
```
<ZIP_ROOT>/<temp>.zip  (25C.zip, 33C.zip, 40C.zip)
  WMXX/Debugger/CPMG/<scan_index>/data.2d
  WMXX/Debugger/CPMG/<scan_index>/acqu.par
```

Sample–WM number mapping:

| Temp | EDA | DAP | DAB |
|------|-----|-----|-----|
| 25C  | WM39 | WM38 | WM40 |
| 33C  | WM66 | WM63 | WM65 |
| 40C  | WM58 | WM54 | WM53 |

DAP2 (2026 repeat of DAP) lives in a separate zip with a different internal structure:

```
DAP2.zip
  Epoxy2026/13DAP/CPMG_25C/<n>/data.2d
  Epoxy2026/13DAP/CPMG_33C/<n>/data.2d
  Epoxy2026/13DAP/CPMG_40C_2/<n>/data.2d
```

Timestamps come from `data.2d` ZipInfo mtime (2-second resolution; sufficient for minute-spaced acquisitions).

### Alpha and T2(alpha) model

Conversion is defined as:

```
alpha = 1 - T2 / T2_0
```

where T2_0 is the T2 of the first non-dropped scan in each series. Alpha should begin at 0 and approach 1 as the epoxy cures.

**TODO: verify alpha scaling** — check alpha vs. elapsed time per sample (should start at 0, increase monotonically, reach a physically reasonable plateau). Confirm T2_0 corresponds to the uncured state (first scan).

**TODO: include DAP2 in all alpha and T2(alpha) plots and verify its alpha vs. t behaviour matches DAP.**

T2 vs. alpha is then fit to Corezzi Eq. 6:

```
T2(alpha) = T2_0 * exp(B * alpha / (a_0 - alpha))
```

where B (negative) controls the rate of T2 decrease and a_0 is the singularity point (gelation/vitrification). The fit returns B, a_0, and their uncertainties, saved to `cpmg_fit_results/t2_alpha_fits.csv`.

**TODO: validate T2(alpha) fits** — check plots in `cpmg_fit_results/t2_alpha/`, confirm a_0 > max(alpha) for all samples, and assess whether late-time dropped scans affect the alpha scaling.
