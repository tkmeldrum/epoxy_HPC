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

**Drop criteria:** A scan is marked `dropped=True` if T2 relative uncertainty > 200% or beta > 2. Dropped scans are excluded from alpha and T2(alpha) analysis but retained in the CSV output. Summary plots show T2 on a log scale (1×10⁻⁵ – 0.05 s) and beta on a linear scale (0–2.1); out-of-range points are marked with an asterisk.

**Adaptive tail averaging:** After a Pass 1 individual fit, the first scan where T2 < 20% of T2(0) (the first reliable, well-constrained T2) defines the start of the low-SNR tail. T2(0) is taken from the first non-dropped scan with T2 relative uncertainty < 50%, making it robust to non-monotonic T2 trajectories (e.g. initial rise before cure-driven decrease). All scans from that point onward are re-binned into ~20 averaged points; scans are only grouped with others of the same echo count. The remainder in each echo-count run is folded into the last bin. Averaged regions are shaded in summary plots.

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

Conversion α is defined as:

```
alpha = 1 - T2 / max(T2)
```

where `max(T2)` is the maximum T2 among reliable non-dropped scans (T2 relative uncertainty < 50%) — the most liquid-like state, regardless of when it occurs. Using only reliable scans prevents a spurious late-time fit from being selected as the reference. This handles non-monotonic T2 trajectories (initial rise before cure-driven decrease). Only scans at or after the `max(T2)` point are assigned alpha; pre-peak scans are excluded from downstream analysis.

R2 = 1/T2 vs. alpha is then fit to the Corezzi model:

```
R2(alpha) = R2_0 * exp(B * alpha / (a_0 - alpha))
```

where `R2_0 = 1/max(T2)` (the alpha=0 reference, most liquid state), B (positive) controls the rate of R2 increase, and a_0 is the singularity point (gelation/vitrification). Equivalently, the normalised fit is `R2/R2_0 = T2_0/T2`. The fit returns B, a_0, and their uncertainties, saved to `cpmg_fit_results/t2_alpha_fits.csv`.

Alpha-vs-time plots are always saved to `cpmg_fit_results/t2_alpha/` regardless of fit success. When the fit succeeds, a second panel showing R2/R2_0 vs. alpha with the model curve is added to the same figure.

### Observed trends (Mar 2026 dataset)

**B decreases with temperature** across almost all samples — higher cure temperature produces a more gradual R2 increase per unit conversion, consistent with the system being farther from its glass transition throughout cure.

**a₀ decreases with temperature** consistently — the R2 singularity (vitrification/gelation) occurs at lower conversion at higher temperature, because Tg approaches T_cure sooner.

**By hardener**, EDA has the highest B at 25 and 33°C (most abrupt mobility loss per unit conversion). DAP2 consistently has the lowest B and a₀ at every temperature, notably different from DAP despite being nominally the same hardener — likely reflects a batch or sample-preparation difference and should be investigated.

**T2_0** increases with temperature as expected (more liquid-like uncured state). EDA 40°C is anomalous (T2_0 ≈ 0.112 s, ~3× other samples at 40°C), suggesting either a slower cure onset or that the first reliable scan was captured unusually early.
