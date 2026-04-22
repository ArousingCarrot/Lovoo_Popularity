"""
06_sensitivity.py — PCA target robustness check
 
Tests whether feature-importance rankings are stable across 6 targets:
  0. PC1 popularity score (baseline)
  1. Equal-weight sum of log-engagement metrics
  2–5. Each raw log-engagement metric individually
 
Run from project root:
    python notebooks/06_sensitivity.py
 
Outputs (all in results/sensitivity/):
  - sensitivity_metrics.csv       — r, MAE, RMSE for each model × target
  - sensitivity_rank_corr.csv     — Spearman rank correlation of feature importance vs PC1
  - sensitivity_top10_ranks.csv   — top-10 feature ranks per target (OLS + XGB)
  - figures/sensitivity_rank_heatmap.png
  - figures/sensitivity_importance_grid.png
"""
 
import json
import warnings
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
 
warnings.filterwarnings("ignore")
 
# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "sensitivity"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
 
RANDOM_STATE = 42
TEST_SIZE = 0.2
 
# ── Load ──────────────────────────────────────────────────────────────────────
X = pd.read_csv(DATA_DIR / "X.csv")
y_raw = pd.read_csv(DATA_DIR / "y_raw.csv")   # log1p-transformed engagement cols
y_pc1 = pd.read_csv(DATA_DIR / "y.csv")["popularity_score"]
 
ENG_COLS = ["counts_profileVisits", "counts_kisses", "counts_fans", "counts_g"]
FEATURE_NAMES = X.columns.tolist()
 
# ── Build all 6 targets ───────────────────────────────────────────────────────
targets: dict[str, pd.Series] = {
    "PC1 (baseline)": y_pc1,
    "Equal-weight sum": y_raw[ENG_COLS].sum(axis=1),
}
for col in ENG_COLS:
    short = col.replace("counts_", "")
    targets[short] = y_raw[col]
 
print(f"Targets: {list(targets.keys())}")
print(f"X: {X.shape}\n")
 
# ── Fixed train/test split (same across all targets) ─────────────────────────
idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
 
# ── Best XGB params (use sensible defaults; swap in grid-searched params if available) ──
xgb_params_path = ROOT / "results" / "xgb_best_params.json"
if xgb_params_path.exists():
    with open(xgb_params_path) as f:
        xgb_kwargs = json.load(f)
    print(f"Loaded XGB params from {xgb_params_path}")
else:
    xgb_kwargs = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1, reg_alpha=0)
    print("Using default XGB params (run 05_xgboost.py first for tuned params)")
 
# ── Helper: metrics ───────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return {
        "pearson_r": round(float(r), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
    }
 
# ── Helper: OLS importance (standardised |coef|) ──────────────────────────────
def ols_importance(X_tr, y_tr):
    m = LinearRegression().fit(X_tr, y_tr)
    return pd.Series(np.abs(m.coef_), index=FEATURE_NAMES)
 
# ── Helper: XGB gain importance ───────────────────────────────────────────────
def xgb_importance(X_tr, y_tr):
    m = XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
        **xgb_kwargs,
    )
    m.fit(X_tr, y_tr)
    return pd.Series(m.feature_importances_, index=FEATURE_NAMES), m
 
# ── Main loop ─────────────────────────────────────────────────────────────────
all_metrics   = []   # rows: (target, model, r, mae, rmse)
ols_imps      = {}   # target -> importance Series
xgb_imps      = {}   # target -> importance Series
 
for tname, y_full in targets.items():
    y_tr = y_full.iloc[train_idx].values
    y_te = y_full.iloc[test_idx].values
 
    # OLS
    ols_imp = ols_importance(X_train, y_tr)
    pred_ols = LinearRegression().fit(X_train, y_tr).predict(X_test)
    m_ols = metrics(y_te, pred_ols)
    ols_imps[tname] = ols_imp
    all_metrics.append({"target": tname, "model": "OLS", **m_ols})
 
    # XGBoost
    xgb_imp, xgb_model = xgb_importance(X_train, y_tr)
    pred_xgb = xgb_model.predict(X_test)
    m_xgb = metrics(y_te, pred_xgb)
    xgb_imps[tname] = xgb_imp
    all_metrics.append({"target": tname, "model": "XGBoost", **m_xgb})
 
    print(f"[{tname}]  OLS r={m_ols['pearson_r']}  XGB r={m_xgb['pearson_r']}")
 
# ── Save metrics table ────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(RESULTS_DIR / "sensitivity_metrics.csv", index=False)
print(f"\nSaved metrics → {RESULTS_DIR / 'sensitivity_metrics.csv'}")
 
# ── Rank correlation vs PC1 (Spearman) ───────────────────────────────────────
# For each model type, compute Spearman r between feature-importance ranks
# for each alternate target vs PC1.
 
rank_rows = []
pc1_name = "PC1 (baseline)"
 
for model_label, imp_dict in [("OLS", ols_imps), ("XGBoost", xgb_imps)]:
    base_ranks = imp_dict[pc1_name].rank(ascending=False)
    for tname, imp in imp_dict.items():
        alt_ranks = imp.rank(ascending=False)
        rho, pval = spearmanr(base_ranks, alt_ranks)
        rank_rows.append({
            "model": model_label,
            "target": tname,
            "spearman_rho": round(rho, 4),
            "p_value": round(pval, 4),
        })
 
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(RESULTS_DIR / "sensitivity_rank_corr.csv", index=False)
print(f"Saved rank correlations → {RESULTS_DIR / 'sensitivity_rank_corr.csv'}")
 
# ── Top-10 feature ranks per target ──────────────────────────────────────────
top_rows = {}
for tname in targets:
    top_rows[f"OLS|{tname}"]     = ols_imps[tname].rank(ascending=False).astype(int)
    top_rows[f"XGB|{tname}"]     = xgb_imps[tname].rank(ascending=False).astype(int)
 
ranks_wide = pd.DataFrame(top_rows, index=FEATURE_NAMES)
ranks_wide.to_csv(RESULTS_DIR / "sensitivity_top10_ranks.csv")
print(f"Saved rank table → {RESULTS_DIR / 'sensitivity_top10_ranks.csv'}")
 
# ── Figure 1: Spearman rank-correlation heatmap ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (model_label, imp_dict) in zip(axes, [("OLS", ols_imps), ("XGBoost", xgb_imps)]):
    target_names = list(imp_dict.keys())
    n = len(target_names)
    rho_matrix = np.ones((n, n))
    for i, t1 in enumerate(target_names):
        r1 = imp_dict[t1].rank(ascending=False)
        for j, t2 in enumerate(target_names):
            r2 = imp_dict[t2].rank(ascending=False)
            rho_matrix[i, j], _ = spearmanr(r1, r2)
    sns.heatmap(
        rho_matrix, xticklabels=target_names, yticklabels=target_names,
        annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title(f"{model_label} — Feature-Rank Spearman ρ across Targets")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
 
plt.tight_layout()
plt.savefig(FIG_DIR / "sensitivity_rank_heatmap.png", dpi=150)
plt.close()
print(f"Saved heatmap → {FIG_DIR / 'sensitivity_rank_heatmap.png'}")
 
# ── Figure 2: Feature importance bar grid (top-10 per target, both models) ───
TARGET_NAMES = list(targets.keys())
N_TOP = 10
fig, axes = plt.subplots(
    2, len(TARGET_NAMES),
    figsize=(4 * len(TARGET_NAMES), 9),
    sharey=False,
)
 
for col_idx, tname in enumerate(TARGET_NAMES):
    for row_idx, (model_label, imp_dict) in enumerate([("OLS", ols_imps), ("XGBoost", xgb_imps)]):
        ax = axes[row_idx][col_idx]
        imp = imp_dict[tname].sort_values(ascending=False).head(N_TOP)
        ax.barh(imp.index[::-1], imp.values[::-1], color="steelblue")
        ax.set_title(f"{model_label}\n{tname}", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
 
plt.suptitle("Top-10 Feature Importances across Targets", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "sensitivity_importance_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved importance grid → {FIG_DIR / 'sensitivity_importance_grid.png'}")
 
# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SENSITIVITY SUMMARY")
print("=" * 60)
print("\nPearson r across targets:\n")
print(metrics_df.pivot(index="target", columns="model", values="pearson_r").to_string())
 
print("\nSpearman ρ of feature ranks vs PC1 (baseline):\n")
summary = rank_df[rank_df["target"] != pc1_name][["model", "target", "spearman_rho"]]
print(summary.to_string(index=False))
 
print(f"""
Interpretation guide:
  ρ > 0.90  → rankings are very stable; PC1 is a clean summary
  ρ 0.70–0.90 → mostly stable with minor divergence; PC1 is still fine
  ρ < 0.70  → notable divergence; discuss in report which features shift
""")