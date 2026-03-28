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

## Kinetic fitting of NMR alpha(t)

### Kamal-Malkin (`fit_nmr_km.py`)

Fits the NMR-derived α(t) curves to the Kamal-Malkin ODE:

```
dα/dt = (k1 + k2·α^m) · (1-α)^(n/2) · (r-α)^(n/2)
```

**r = 2.0 fixed** by stoichiometry for all samples. Time is converted from minutes to **seconds** before fitting so that k1 and k2 are in s⁻¹, consistent with DSC-derived parameters in `fit_results/combined_results.csv`.

Parameters: `[log_k1, log_k2, m, n, log_sigma]` (5 free). Fitting uses Nelder-Mead least-squares; optional MCMC via `--mcmc` flag (emcee, 64 walkers). Outputs: `fit_results_nmr/km_results.csv`. Per-dataset plots (data + fit + annotated parameters) saved to `fit_plots_nmr/`. MCMC chains saved to `mcmc_samples_nmr/` as `*_fitdata.npz` for post-processing with `plot_mcmc.py`.

**Comparison to DSC fits (Mar 2026, NMR 25°C datasets):** after converting NMR time to seconds, k2 values agree with DSC to within a factor of ~2 for most samples (e.g. EDA 25°C: NMR 2.6×10⁻⁴ s⁻¹ vs DSC 2.7×10⁻⁴ s⁻¹). Residual differences reflect the different r used (r=2 here vs r≈0.6–0.8 in DSC fits) and the different alpha scales (NMR alpha from T2, DSC alpha from enthalpy). k1 is frequently negligible in NMR fits (hits lower bound), consistent with the autocatalytic regime dominating at low temperature. **EDA 40°C is an outlier** — RSS is ~50× larger than all other datasets, consistent with the anomalous T2_0 for that sample; the KM model fails to capture this cure trajectory.

### Corezzi diffusion-corrected kinetics (`BatchBayesian_nmr_corezzi.bak` — archived)

Extends the KM model with diffusion correction (Corezzi Eq. 8):

```
k_eff_i = k_ci / (1 + (k_ci/k0) · exp(ξ · B · α / (a0 - α)))
dα/dt   = (k_eff1 + k_eff2·α^m) · (1-α)^(n/2) · (r-α)^(n/2)
```

B and a0 are **fixed** per (sample, temp) from the NMR R2(α) fit (`cpmg_fit_results/t2_alpha_fits.csv`). Additional free parameters: ξ (dimensionless diffusion coupling) and log_k0 (reference rate). As ξ→0 the model reduces to Kamal-Malkin. Outputs: `fit_results_nmr/corezzi_results.csv`, with RSS printed alongside KM RSS for direct comparison.

**Graphical output — LS fit (always, `*_corezzi_ls.png`):** 3-panel figure saved to `fit_plots_nmr/`:
1. α(t) — data points, Corezzi fit, KM overlay (if `km_results.csv` present). Parameter box with kc1, kc2, m, n, ξ, k0 ± Laplace uncertainties and fixed B, a₀.
2. dα/dt vs α — data (numerical gradient), Corezzi rate curve, KM rate curve.
3. k_eff(α) on log scale — k_eff1(α) and k_eff2(α) showing diffusion suppression toward zero as α → a₀; bare kc1, kc2 shown as dashed references; a₀ marked with a vertical line.

**Graphical output — MCMC (with `--mcmc`, `*_corezzi_combined.png` + `*_corezzi_corner.png`):** same 3-panel layout with posterior median and 95% CI bands throughout, including CI bands on the k_eff(α) curves. Corner plot saved separately.

---

## Script usage

All scripts should be run from the repo root. Machine-specific paths are set in `local_config.py`.

### `cpmg_batch_fit.py` — CPMG NMR processing

```bash
python cpmg_batch_fit.py                          # process all samples, all temperatures
python cpmg_batch_fit.py --diagnose EDA 40C       # plot raw decay diagnostics for one dataset
```

Outputs to `cpmg_fit_results/`: `all_samples.csv`, per-sample CSVs, summary plots, T2(α) plots, and `t2_alpha_fits.csv`.

---

### `fit_nmr_km.py` — Kamal-Malkin fit to NMR α(t)

```bash
python fit_nmr_km.py                    # LS fit, all NMR datasets
python fit_nmr_km.py EDA 25C            # LS fit, one dataset
python fit_nmr_km.py --mcmc             # LS + MCMC, all datasets
python fit_nmr_km.py --mcmc EDA 25C     # LS + MCMC, one dataset
```

Outputs: `fit_results_nmr/km_results.csv`, per-dataset LS plots in `fit_plots_nmr/`. With `--mcmc`, chains saved to `mcmc_samples_nmr/` as `*_fitdata.npz` for post-processing with `plot_mcmc.py`.

---

### `fit_kuro.py` / `fit_kuro_fixedr.py` — Full MCMC for DSC and NMR (fixed r, from `.mat`)

The primary kinetic fitting scripts for the existing DSC dataset. r is fixed at max(α) per dataset (not stoichiometric).

```bash
python fit_kuro_fixedr.py               # MCMC, all DSC + NMR datasets
python fit_kuro_fixedr.py NMR EDA 25    # MCMC, one dataset (temp as integer)
python fit_kuro_fixedr.py NMR EDA 25 --grid_scan   # posterior grid scan first
```

Outputs: `mcmc_samples/`, `fit_results/fixed_r/`.

---

### `plot_mcmc.py` — Post-process MCMC chains from saved `.npz` files

Generates posterior overlay, α(t) CI band, dα/dt vs α, chain trace, and corner plots, plus a summary grid image. Appends a row to `posterior_summary.csv` (written before plots so results survive a crash). Intended to run locally after transferring results from the cluster via rsync.

```bash
python plot_mcmc.py                                        # all files in mcmc_samples/
python plot_mcmc.py path/to/file_fitdata.npz               # single file
python plot_mcmc.py --r 2.0 --input-dir mcmc_samples_nmr --outdir fit_plots_nmr
python plot_mcmc.py --burnin 5000 --stride 5               # override mcmc_config
```

Only processes `.npz` files directly in `--input-dir` — does not recurse into subdirectories.

**`--summary`:** defaults to `posterior_summary_{timestamp}.csv` so each run produces a uniquely named file. Pass `--summary myfile.csv` to override.

**`--r`:** use `--r 2.0` for NMR/KM data (stoichiometric ratio). Without it, defaults to `max(a_data)` for DSC data.

---

### `MCMC_diagnostics.py` — Chain diagnostics (autocorrelation, acceptance, Gelman-Rubin)

```bash
python MCMC_diagnostics.py                             # all files in mcmc_samples/
python MCMC_diagnostics.py mcmc_samples/NMR_EDA_25C_fitdata.npz      # one file
```

---

### `run_corezzi_all.sh` — Run all 12 NMR Corezzi MCMC fits sequentially (local)

Runs all 12 (sample, temp) combinations back-to-back on the local machine. Intended for overnight runs on a laptop.

```bash
conda activate epoxy
mkdir -p logs
nohup ./run_corezzi_all.sh > logs/corezzi_overnight.log 2>&1 &
tail -f logs/corezzi_overnight.log   # monitor progress
```

---

### `epoxy_corezzi_nmr.sh` — SLURM array job for bora cluster

Submits all 12 NMR Corezzi MCMC fits as a 12-element SLURM array on bora (20 cores/node, 72 h walltime). Each task runs one (sample, temp) independently.

```bash
mkdir -p /sciclone/home/tkmeldrum/epoxy_kinetics/logs
sbatch epoxy_corezzi_nmr.sh
squeue -u tkmeldrum --start   # check estimated start time
```

**Before submitting:** ensure `local_config.py` on the cluster has `N_WORKERS = 20`, and that `mcmc_config.py` has `nsteps > burnin` (e.g. `nsteps = 30000`, `burnin = 5000`). Copy `cpmg_fit_results/t2_alpha_fits.csv`, `cpmg_fit_results/all_samples.csv`, and optionally `fit_results_nmr/km_results.csv` to the cluster before running.

**Note on parallelism:** `N_WORKERS` in `local_config.py` controls the `multiprocessing.Pool` size and should match `--cpus-per-task`. The natural ceiling for emcee parallelism is `nwalkers / 2`; requesting more cores than this wastes allocation. On bora with `nwalkers = 64`, the effective ceiling is 32 — but bora nodes only have 20 cores, so `N_WORKERS = 20` is the practical limit.

---

### `final_results_to_plots.py` — Arrhenius analysis and parameter trend plots

Reads a posterior summary CSV and produces two PDFs saved to `results/`:

1. **`results/fit_trends_{timestamp}.pdf`** — all KM parameters (k₁, k₂, m, n, r) vs 1/T for each sample and method, with error bars from posterior CIs.
2. **`results/arrhenius_fits_{timestamp}.pdf`** — ln(k₁) and ln(k₂) vs 1/T with linear fits and printed activation energies (Ea ± uncertainty in kJ/mol).

```bash
python final_results_to_plots.py                              # uses posterior_summary.csv
python final_results_to_plots.py combined_arrhenius_DATE.csv  # specify input file
```

**Input CSV format:** `Label` column with `{Method}_{Sample}_{Temp}C` (e.g. `DSC_EDA_25C`), plus `log_k1_median/CI_lower/CI_upper`, `log_k2_median/CI_lower/CI_upper`, `m_median/CI_lower/CI_upper`, `n_median/CI_lower/CI_upper`. DAP2 is automatically remapped to DAP/NMR2 for plotting.

**NMR LS fits have no posterior CI** — CI columns are set equal to the median, giving zero-width error bars and unweighted Arrhenius regression.

---

### `combined_arrhenius_{DATE}.csv` — merged DSC + NMR KM results

Combined input file for `final_results_to_plots.py`, merging:
- DSC MCMC posteriors from `posterior_summary.csv` (18 datasets, 25–100°C)
- NMR KM LS fits from `fit_results_nmr/km_results.csv` (12 datasets, 25–40°C)

Regenerate by re-running the merge inline or from a script if either source file is updated.
