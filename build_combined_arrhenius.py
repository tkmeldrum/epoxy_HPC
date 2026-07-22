#!/usr/bin/env python3
"""build_combined_arrhenius.py
-----------------------------
Merge DSC and NMR posterior summaries into a single combined_arrhenius_{date}.csv
for final_results_to_plots.py.

Always writes a new dated file -- never overwrites a previous run's output.

Run from the repo root:
    python build_combined_arrhenius.py
"""
import os
from datetime import datetime

import pandas as pd

DSC_CSV = "posterior_summary_DSC.csv"
NMR_CSV = "posterior_summary_NMR_fixedr.csv"

dsc = pd.read_csv(DSC_CSV)
nmr = pd.read_csv(NMR_CSV)

# NMR rows have no r_* columns (r is fixed = max(alpha) per dataset, not fitted);
# concat aligns on shared columns and fills the rest with NaN.
combined = pd.concat([dsc, nmr], ignore_index=True)

date_str = datetime.now().strftime("%Y%m%d")
out_path = f"combined_arrhenius_{date_str}.csv"
if os.path.exists(out_path):
    raise FileExistsError(
        f"{out_path} already exists -- refusing to overwrite. "
        "Delete it first if you really want to regenerate today's file."
    )

combined.to_csv(out_path, index=False)
print(f"Wrote {len(combined)} rows ({len(dsc)} DSC + {len(nmr)} NMR) to {out_path}")
