"""Generate the sand/shale dataset and plot how the features relate to the class (Task 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42                     # fixed random seed: same data every run
N = 1000                      # number of depth samples
P_SAND = 0.20                 # target share of class 1 (sand)
rng = np.random.default_rng(SEED)

# ---- class label y: 1 = sand (fluid), 0 = shale (tight, no fluid) ------
y = rng.binomial(1, P_SAND, N)
sand, shale = y == 1, y == 0
n1, n0 = sand.sum(), shale.sum()

# ---- sand split by hydrogen index (HI) for the two f2 modes -----------
low_hi = np.zeros(N, bool)
low_hi[sand] = rng.random(n1) < 0.40              # 40 % of sand: low HI
high_hi = sand & ~low_hi                          # 60 % of sand: high HI

# ---- f1: porosity, Gaussian per class, different means ----------------
f1 = np.empty(N)
f1[shale] = rng.normal(0.03, 0.015, n0)           # tight: phi ~ 0
f1[sand]  = rng.normal(0.20, 0.07, n1)
f1 = np.clip(f1, 0, None)

# ---- f2: hydrogen index HI of the pore fluid --------------------------
# shale: one mode (no pore fluid, HI ~ 0)
# sand : two well-separated modes -> low HI and high HI
f2 = np.empty(N)
f2[shale] = rng.normal(0.03, 0.015, n0)           # tight
f2[low_hi]  = rng.normal(0.25, 0.06, low_hi.sum())   # low HI:  0 - 0.5
f2[high_hi] = rng.normal(0.97, 0.03, high_hi.sum())  # high HI: ~ 1
f2 = np.clip(f2, 0, None)

# ---- f3, f4: standardised log residuals with a sign rule --------------
# Both are N(0,1) in each class; only the sign of f3*f4 carries the class.
P_RULE = 0.85                                     # 15 % break the rule -> overlap
f3 = rng.normal(0, 1, N)
mag4 = np.abs(rng.normal(0, 1, N))
follow = rng.random(N) < P_RULE
same_sign = np.where(sand, follow, ~follow)       # sand: same sign, shale: opposite
f4 = np.where(same_sign, np.sign(f3), -np.sign(f3)) * mag4

# ---- cat: fluid category created from HI (f2) ------------------------
# HI ranges:  tight < 0.10 <= gas < 0.65 <= oil < 0.975 <= water
# Neighbouring ranges overlap: inside the overlap zone around each
# boundary, the row gets one of the two neighbouring labels at random.
levels = np.array(["tight", "gas", "oil", "water"])
edges  = np.array([0.10, 0.65, 0.975])            # boundaries between categories
half_w = np.array([0.04, 0.10, 0.02])             # half-width of each overlap zone

idx = np.searchsorted(edges, f2)                  # label from HI without overlap
for k, edge in enumerate(edges):
    zone = np.abs(f2 - edge) < half_w[k]          # rows inside overlap zone k
    idx[zone] = k + rng.integers(0, 2, zone.sum())  # random: level k or k+1
cat = levels[idx].astype(object)

# ---- shale class: one category only -------------------------------------
cat[shale] = "tight"

# ---- sand class: porosity categories (f1) replace the fluid categories --
#   poor   : phi < 0.10
#   medium : 0.10 <= phi < 0.25
#   good   : 0.25 <= phi <= 0.40
sand_levels = np.array(["poor", "medium", "good"], dtype=object)
cat[sand] = sand_levels[np.searchsorted([0.10, 0.25], f1[sand], side="right")]

MAX_RARE = 5                                      # rule: tight in sand <= 5 rows
rare = np.where(sand & (cat == "tight"))[0]
cat[rare[MAX_RARE:]] = "medium"                   # keep at most 5

# ---- assemble and save -------------------------------------------------
df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4,
                   "cat": cat, "y": y})

OUT_DIR = Path(__file__).resolve().parent      # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)        # create folder if missing
out_file = OUT_DIR / "sand_shale_dataset.csv"
df.to_csv(out_file, index=False, float_format="%.5f")
print(f"Saved {N} rows to {out_file}, sand share = {df.y.mean():.3f}")

# ---- check the feature rules -------------------------------------------
same = np.sign(f3) == np.sign(f4)
print("f1 mean shale / sand :", f1[shale].mean().round(3), f1[sand].mean().round(3))
print("f2 (HI) mean by cat:", df.groupby("cat").f2.mean().round(3).to_dict())
print("f3,f4 same-sign share shale / sand:",
      same[shale].mean().round(3), same[sand].mean().round(3))
print(pd.crosstab(df.cat, df.y))
assert (sand & (cat == "tight")).sum() <= MAX_RARE

# ---- plot data -----------------------------------------------------------
df_shale, df_sand = df[df.y == 0], df[df.y == 1]
C0, C1 = "#2a78d6", "#eb6834"                     # colour: y = 0 shale, y = 1 sand
LAB0, LAB1 = "y = 0 (shale)", "y = 1 (sand)"

# ---- table 1: class balance ---------------------------------------------
balance = df.y.value_counts().sort_index().rename("count").to_frame()
balance.index = ["y = 0 (shale)", "y = 1 (sand)"]
balance["percent"] = (100 * balance["count"] / len(df)).round(1)
balance.loc["total"] = [balance["count"].sum(), 100.0]
balance["count"] = balance["count"].astype(int)

# ---- table 2: category counts per class ---------------------------------
cat_counts = pd.crosstab(df.cat, df.y, margins=True, margins_name="total")
cat_counts.columns = ["y = 0 (shale)", "y = 1 (sand)", "total"]
cat_counts = cat_counts.reindex(["tight", "poor", "medium", "good", "total"])

print("Class balance\n", balance, "\n")
print("Category counts per class\n", cat_counts, "\n")
balance.to_csv(OUT_DIR / "table_class_balance.csv")
cat_counts.to_csv(OUT_DIR / "table_cat_counts.csv")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))

# ---- helper: histogram of one feature per class ------------------------
def hist_by_class(axis, col, title, xlabel, bins=40):
    """Plot the histogram of one feature for both classes on the same axis.

    Parameters
    ----------
    axis : matplotlib Axes
        Subplot to draw on.
    col : str
        Name of the feature column in df.
    title : str
        Subplot title.
    xlabel : str
        Label of the x-axis.
    bins : int, optional
        Number of histogram bins (default 40).
    """
    bin_edges = np.histogram_bin_edges(df[col], bins=bins)
    axis.hist(df_shale[col], bin_edges, density=True, alpha=0.6, color=C0, label=LAB0)
    axis.hist(df_sand[col],  bin_edges, density=True, alpha=0.6, color=C1, label=LAB1)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("density")
    axis.legend()

# ---- f1: porosity (Gaussian per class, different means) -----------------
hist_by_class(ax[0, 0], "f1", "f1: porosity", "porosity (fraction)")

# ---- f2: hydrogen index (shale one mode, sand two modes) ----------------
hist_by_class(ax[0, 1], "f2", "f2: hydrogen index (HI)", "HI")

# ---- f3 and f4: same distribution in both classes -----------------------
hist_by_class(ax[0, 2], "f3", "f3: same distribution in both classes", "f3")
hist_by_class(ax[1, 0], "f4", "f4: same distribution in both classes", "f4")

# ---- f3 vs f4: sign rule (sand same sign, shale opposite sign) ---------
a = ax[1, 1]
a.scatter(df_shale.f3, df_shale.f4, s=10, alpha=0.5, color=C0, label=LAB0)
a.scatter(df_sand.f3,  df_sand.f4,  s=10, alpha=0.7, color=C1, label=LAB1)
a.axhline(0, color="0.5", lw=0.8)
a.axvline(0, color="0.5", lw=0.8)
a.set_title("f3 vs f4: sign rule")
a.set_xlabel("f3")
a.set_ylabel("f4")
a.legend()

# ---- cat: level counts per class ----------------------------------------
a = ax[1, 2]
counts = pd.crosstab(df.cat, df.y)
x = np.arange(len(counts))
BAR_W = 0.4
a.bar(x - BAR_W / 2, counts[0], BAR_W, color=C0, label=LAB0)
a.bar(x + BAR_W / 2, counts[1], BAR_W, color=C1, label=LAB1)
for i, (c0, c1) in enumerate(zip(counts[0], counts[1])):  # count on each bar
    a.text(i - BAR_W / 2, c0, c0, ha="center", va="bottom", fontsize=9)
    a.text(i + BAR_W / 2, c1, c1, ha="center", va="bottom", fontsize=9)
a.set_xticks(x, counts.index)
a.set_title("cat: level counts per class")
a.set_ylabel("count")
a.legend()

# ---- save ---------------------------------------------------------------
plt.tight_layout()
out_file = OUT_DIR / "explore_features.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to {out_file}")
plt.show()
