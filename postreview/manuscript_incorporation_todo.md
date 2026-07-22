# Manuscript incorporation TODO — 60°C NMR data

Redirect these `.tex` files to point at `postreview/pub_figs_60C/` instead of
`pub_figs/figures/` / the live `Ea_table.tex`. Only 3 figures and 2 `\input`s
actually differ with the 60°C data — everything else is untouched, don't
bother redirecting it.

## `main.tex`, `redline.tex`, `rev1.tex`

Same 3 lines in each file, just different line numbers per file:

| File | `parameters.pdf` | `arrhenius.pdf` | `ea_bars.pdf` |
|---|---|---|---|
| `main.tex` | line 170 | line 191 | line 199 |
| `redline.tex` | line 253 | line 274 | line 282 |
| `rev1.tex` | line 176 | line 197 | line 205 |

Change:
```
.../pub_figs/figures/parameters.pdf  ->  .../epoxy_HPC/postreview/pub_figs_60C/parameters.pdf
.../pub_figs/figures/arrhenius.pdf   ->  .../epoxy_HPC/postreview/pub_figs_60C/arrhenius.pdf
.../pub_figs/figures/ea_bars.pdf     ->  .../epoxy_HPC/postreview/pub_figs_60C/ea_bars.pdf
```

## `SI.tex`

- Line 85: `\input{.../Epoxy-Kinetics-2025/Ea_table.tex}`
  → `\input{.../epoxy_HPC/postreview/pub_figs_60C/Ea_table.tex}`
- Line 87: `\input{.../pub_figs/figures/si_figures.tex}`
  → `\input{.../epoxy_HPC/postreview/pub_figs_60C/si_figures.tex}`
  (pulls in the KM table and all SI figures, including the new 60°C ones, in one shot)

## Not referenced anywhere currently — nothing to do

`eyring.pdf` and `Ea_only_table.tex` aren't `\input`/`\includegraphics`'d in any
manuscript file right now, so there's nothing to redirect for them.

## `cpmg_fit.pdf` / `representative.pdf` — leave alone

These two are **unaffected** by the 60°C work — they still render their original
default single-dataset example (not something that changes with more data), so
redirecting them would just point at identical content.

## Before finalizing: caption text

`eyring_analysis.py`'s `Ea_table.tex` caption hardcodes the NMR temperature
range as text: `\qtylist{25;33;40}{\celsius}`. This is now wrong (NMR goes to
60°C). Two options:
- Manually edit the caption text in the redirected `postreview/pub_figs_60C/Ea_table.tex`
  before submission, or
- Fix the hardcoded string in `pub_figs/eyring_analysis.py` (`write_latex_table`/
  `write_ea_only_table`, two occurrences) and regenerate.

Either way, don't let the caption silently claim NMR stops at 40°C once the
table shows 60°C rows.

## After editing the `.tex` files

Recompile the manuscript and visually check the KM table (`table_km.tex`) —
the DAP NMR block spans a `\multirow` that changes size when 60°C data is
present (see `postreview/datachanges.md` for why), so it's worth a visual
sanity check that the table renders without misaligned rows.
