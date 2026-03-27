import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

DATA_DIR = "data/processed"
FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Load log-transformed engagement ───────────────────────────────────────
y_raw = pd.read_csv(f"{DATA_DIR}/y_raw.csv")
ENG_COLS = ["counts_profileVisits", "counts_kisses", "counts_fans", "counts_g"]
print(f"y_raw shape: {y_raw.shape}")
print(y_raw.describe())

# ── 2. Standardize before PCA ─────────────────────────────────────────────────
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y_raw[ENG_COLS])

# ── 3. Fit PCA ────────────────────────────────────────────────────────────────
pca = PCA(n_components=4)
pca.fit(y_scaled)

explained = pca.explained_variance_ratio_
print(f"\nExplained variance per component: {explained.round(3)}")
print(f"PC1 explains {explained[0]*100:.1f}% of variance")

loadings = pd.Series(pca.components_[0], index=ENG_COLS)
print(f"\nPC1 loadings:\n{loadings.round(4)}")

# ── 4. Verify all loadings positive ──────────────────────────────────────────
if (loadings > 0).all():
    print("\nAll PC1 loadings are positive -- PC1 is a valid popularity signal.")
else:
    print("\nWARNING: Not all PC1 loadings are positive. Check signs before proceeding.")
    # Flip sign if PC1 is inverted (arbitrary sign convention in PCA)
    if (loadings < 0).all():
        print("All loadings negative -- flipping sign of PC1.")
        pca.components_[0] *= -1
        loadings = pd.Series(pca.components_[0], index=ENG_COLS)
        print(f"Flipped PC1 loadings:\n{loadings.round(4)}")

# ── 5. Compute popularity score ───────────────────────────────────────────────
y_pca = pca.transform(y_scaled)
y = pd.DataFrame({
    "popularity_score": y_pca[:, 0],
    "PC2": y_pca[:, 1],
    "PC3": y_pca[:, 2],
    "PC4": y_pca[:, 3],
})
print(f"\nPopularity score stats:\n{y['popularity_score'].describe()}")

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scree plot
axes[0].bar(range(1, 5), explained * 100)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("Scree Plot")
axes[0].set_xticks(range(1, 5))

# PC1 loadings
loadings.plot.barh(ax=axes[1], color=["steelblue" if v > 0 else "tomato" for v in loadings])
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("PC1 Loadings")
axes[1].set_xlabel("Loading")

# Popularity score distribution
y["popularity_score"].hist(bins=50, ax=axes[2])
axes[2].set_title("Popularity Score Distribution")
axes[2].set_xlabel("PC1 score")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/pca_loadings.png", dpi=150)
plt.show()

# ── 7. Save ───────────────────────────────────────────────────────────────────
y.to_csv(f"{DATA_DIR}/y.csv", index=False)
print(f"\nSaved popularity scores to {DATA_DIR}/y.csv")
print("PCA complete.")