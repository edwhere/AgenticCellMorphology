"""Generate data/sample.csv: synthetic cell-morphology measurements.

`area` and `perimeter` are deliberately strongly correlated (so the two-run
branch is exercised), `nucleus_area` tracks `area` too, and `phenotype` depends
on area, circularity, intensity and the categorical `cell_line`.

    uv run python scripts/make_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 600

area = rng.lognormal(mean=6.0, sigma=0.35, size=n)
perimeter = 2 * np.sqrt(np.pi * area) * rng.normal(1.08, 0.04, n)
nucleus_area = area * rng.normal(0.3, 0.025, n)
circularity = np.clip(rng.beta(8, 2, n), 0, 1)
mean_intensity = rng.normal(120, 25, n)
texture_contrast = rng.gamma(4, 2, n)
cell_line = rng.choice(["HeLa", "U2OS", "MCF7"], n, p=[0.4, 0.35, 0.25])

z = (
    1.8 * (np.log(area) - 6.0) / 0.35
    - 1.2 * (circularity - 0.8) / 0.12
    + 0.6 * (mean_intensity - 120) / 25
    + np.select([cell_line == "MCF7", cell_line == "U2OS"], [0.9, -0.4], 0.0)
    + rng.normal(0, 1.0, n)
)
phenotype = np.where(z > 0.3, "abnormal", "normal")

df = pd.DataFrame({
    "cell_id": np.arange(1, n + 1),
    "area": area.round(2),
    "perimeter": perimeter.round(2),
    "nucleus_area": nucleus_area.round(2),
    "circularity": circularity.round(4),
    "mean_intensity": mean_intensity.round(2),
    "texture_contrast": texture_contrast.round(3),
    "cell_line": cell_line,
    "phenotype": phenotype,
})
# a few missing values to exercise imputation
for col in ["mean_intensity", "texture_contrast"]:
    df.loc[rng.choice(n, 12, replace=False), col] = np.nan

out = Path(__file__).resolve().parents[1] / "data" / "sample.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
print(df["phenotype"].value_counts().to_string())
