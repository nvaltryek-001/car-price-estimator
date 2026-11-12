"""Regenerate the README evaluation charts from the dataset and the same split/pipeline as backend/train_model.py.

Usage (from repo root):  python scripts/make_charts.py
Writes PNGs to docs/charts/.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BLUE, ORANGE, GREY = "#2563eb", "#f97316", "#6b7280"

df = pd.read_csv(ROOT / "backend" / "data" / "car_data.csv").drop_duplicates().copy()
df["Age"] = 2026 - df["Year"]
X, y = df.drop(columns=["Selling_Price", "Car_Name"]), df["Selling_Price"]
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {n: joblib.load(ROOT / "backend" / "ml" / f) for n, f in
          [("Linear Regression", "linear_regression_model.joblib"), ("Random Forest", "random_forest_model.joblib")]}

# 1. Predicted vs actual on the held-out test split
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
lim = [0, max(yte.max(), 1) * 1.05]
for ax, (name, m), c in zip(axes, models.items(), [GREY, BLUE]):
    ax.scatter(yte, m.predict(Xte), s=22, alpha=0.75, color=c)
    ax.plot(lim, lim, "--", color="#9ca3af", lw=1)
    ax.set(title=name, xlabel="Actual price (lakh ₹)", xlim=lim, ylim=lim)
axes[0].set_ylabel("Predicted price (lakh ₹)")
fig.suptitle(f"Predicted vs actual — held-out test set (n={len(yte)})", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "predicted_vs_actual.png"); plt.close(fig)

# 2. Random Forest feature importance (aggregated back to original columns)
rf = models["Random Forest"]
names = rf.named_steps["preprocessor"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
imp.index = [n.split("__", 1)[1].split("_")[0] if n.startswith("cat__") else n.split("__", 1)[1] for n in imp.index]
imp = imp.groupby(level=0).sum().sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(imp.index, imp.values, color=BLUE)
ax.set(title="What drives the price? (Random Forest importance)", xlabel="Share of importance")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png"); plt.close(fig)

# 3. EDA: price by car age and fuel type
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for fuel, g in df.groupby("Fuel_Type"):
    axes[0].scatter(g["Age"], g["Selling_Price"], s=16, alpha=0.7, label=fuel)
axes[0].set(title="Selling price vs car age", xlabel="Age (years)", ylabel="Selling price (lakh ₹)"); axes[0].legend(frameon=False)
med = df.groupby("Fuel_Type")["Selling_Price"].median().sort_values()
axes[1].bar(med.index, med.values, color=[GREY, ORANGE, BLUE][: len(med)])
axes[1].set(title="Median selling price by fuel", ylabel="Lakh ₹")
fig.tight_layout(); fig.savefig(OUT / "eda.png"); plt.close(fig)
print("charts ->", OUT)
