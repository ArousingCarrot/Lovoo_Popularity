"""
07_xgboost_robustness.py — Robustness evaluation for the tuned XGBoost model.

Mirrors the robustness checks in 04_baseline_lr_full.py so results are
directly comparable across all three models:
  - 5-fold cross-validation (CV r, CV RMSE)
  - Repeated random splits (5 seeds)
  - Gaussian noise injection (5%, 10%, 20% of feature range)
  - Synthetic outliers (5% of continuous feature values → 3σ above mean)

Requires the saved model: results/xgb_model.joblib
Requires the processed data: data/processed/X.csv, data/processed/y.csv

Run from project root:
    python notebooks/07_xgboost_robustness.py

Outputs (all in results/):
    xgb_robustness_metrics.json   — full robustness results
    xgb_cv_metrics.json           — CV folds + mean/std (mirrors OLS format)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_FOLDS      = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    X = pd.read_csv(DATA_DIR / "X.csv")
    y = pd.read_csv(DATA_DIR / "y.csv")["popularity_score"]
    return X, y


def load_model():
    model_path = RESULTS_DIR / "xgb_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. "
            "Run 05_xgboost.py first to train and save the model."
        )
    return joblib.load(model_path)


def load_best_params():
    params_path = RESULTS_DIR / "xgb_best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    # Fallback to known best params from xgb_metrics.json
    return {
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "subsample": 0.8,
    }


def compute_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"pearson_r": float(r), "mae": float(mae), "rmse": float(rmse)}


def make_model(params):
    return XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
        **params,
    )

# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(X, y, params):
    print(f"\nRunning {N_FOLDS}-fold cross-validation...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    fold_r, fold_rmse, fold_mae = [], [], []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr), start=1):
        model = make_model(params)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        preds = model.predict(X_arr[val_idx])
        m = compute_metrics(y_arr[val_idx], preds)
        fold_r.append(m["pearson_r"])
        fold_rmse.append(m["rmse"])
        fold_mae.append(m["mae"])
        print(f"  Fold {fold_idx}: r={m['pearson_r']:.4f}  "
              f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    return {
        "cv_pearson_r_folds":  fold_r,
        "cv_rmse_folds":       fold_rmse,
        "cv_mae_folds":        fold_mae,
        "cv_pearson_r_mean":   float(np.mean(fold_r)),
        "cv_pearson_r_std":    float(np.std(fold_r, ddof=1)),
        "cv_rmse_mean":        float(np.mean(fold_rmse)),
        "cv_rmse_std":         float(np.std(fold_rmse, ddof=1)),
        "cv_mae_mean":         float(np.mean(fold_mae)),
        "cv_mae_std":          float(np.std(fold_mae, ddof=1)),
    }

# ── Robustness evaluation ─────────────────────────────────────────────────────

def evaluate_robustness(X, y, params):
    print("\nRunning robustness evaluation...")
    results = {}

    continuous_features = ["age", "counts_pictures", "counts_details",
                           "whazzup_len", "lang_count"]
    continuous_features = [c for c in continuous_features if c in X.columns]

    def eval_modified(X_mod, split_seed=RANDOM_STATE):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_mod, y, test_size=TEST_SIZE, random_state=split_seed
        )
        model = make_model(params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        return compute_metrics(y_te.to_numpy(), preds)

    # (a) Repeated random splits ───────────────────────────────────────────────
    seeds = [42, 100, 200, 300, 400]
    split_r, split_mae = [], []
    for seed in seeds:
        m = eval_modified(X, split_seed=seed)
        split_r.append(m["pearson_r"])
        split_mae.append(m["mae"])

    results["repeated_splits"] = {
        "r_mean":   float(np.mean(split_r)),
        "r_std":    float(np.std(split_r, ddof=1)),
        "mae_mean": float(np.mean(split_mae)),
        "mae_std":  float(np.std(split_mae, ddof=1)),
        "r_per_seed": dict(zip(seeds, split_r)),
    }
    print(f"  Repeated splits (5 seeds): "
          f"r={results['repeated_splits']['r_mean']:.4f} "
          f"± {results['repeated_splits']['r_std']:.4f}")

    # (b) Gaussian noise injection ─────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_STATE)
    for noise_pct in [0.05, 0.10, 0.20]:
        X_noisy = X.copy()
        for col in continuous_features:
            col_range = X[col].max() - X[col].min()
            sigma = noise_pct * col_range
            X_noisy[col] = X_noisy[col] + rng.normal(0, sigma, size=len(X))

        m   = eval_modified(X_noisy)
        key = f"noise_{int(noise_pct * 100)}pct"
        results[key] = {"r": m["pearson_r"], "mae": m["mae"], "rmse": m["rmse"]}
        print(f"  Noise ({int(noise_pct * 100):>2d}%):        "
              f"r={m['pearson_r']:.4f}  MAE={m['mae']:.4f}")

    # (c) Synthetic outliers (5% of rows → 3σ above mean) ─────────────────────
    X_outliers = X.copy()
    n_outliers  = int(0.05 * len(X))
    for col in continuous_features:
        outlier_val = X[col].mean() + 3 * X[col].std()
        idx = rng.choice(X.index, size=n_outliers, replace=False)
        X_outliers.loc[idx, col] = outlier_val

    m = eval_modified(X_outliers)
    results["outliers_5pct"] = {"r": m["pearson_r"], "mae": m["mae"], "rmse": m["rmse"]}
    print(f"  Outliers (5%):         "
          f"r={m['pearson_r']:.4f}  MAE={m['mae']:.4f}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and model...")
    X, y = load_data()
    params = load_best_params()
    print(f"  X: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Best params: {params}")

    # Baseline test metrics (re-evaluate for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = make_model(params)
    model.fit(X_train, y_train)
    test_metrics = compute_metrics(y_test.to_numpy(), model.predict(X_test))
    print(f"\n  Test (baseline): r={test_metrics['pearson_r']:.4f}  "
          f"MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    # CV
    cv_results = cross_validate(X, y, params)

    # Robustness
    robustness_results = evaluate_robustness(X, y, params)

    # ── Save ──────────────────────────────────────────────────────────────────
    full_output = {
        "n_samples":      int(X.shape[0]),
        "n_features":     int(X.shape[1]),
        "test_size":      TEST_SIZE,
        "random_state":   RANDOM_STATE,
        "best_params":    params,
        "test_pearson_r": test_metrics["pearson_r"],
        "test_mae":       test_metrics["mae"],
        "test_rmse":      test_metrics["rmse"],
        **cv_results,
        "robustness":     robustness_results,
    }

    with open(RESULTS_DIR / "xgb_robustness_metrics.json", "w") as f:
        json.dump(full_output, f, indent=2)

    # Separate CV file to mirror OLS format exactly
    with open(RESULTS_DIR / "xgb_cv_metrics.json", "w") as f:
        json.dump({
            "model": "XGBoost",
            "n_folds": N_FOLDS,
            **cv_results
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("XGBOOST ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"  Samples  : {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"  Test     : r={test_metrics['pearson_r']:.4f}  "
          f"MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")
    print(f"  CV {N_FOLDS}-fold : r={cv_results['cv_pearson_r_mean']:.4f} "
          f"± {cv_results['cv_pearson_r_std']:.4f}  "
          f"RMSE={cv_results['cv_rmse_mean']:.4f} "
          f"± {cv_results['cv_rmse_std']:.4f}")
    print(f"  Repeated splits : r={robustness_results['repeated_splits']['r_mean']:.4f} "
          f"± {robustness_results['repeated_splits']['r_std']:.4f}")
    print(f"  Noise  5% : r={robustness_results['noise_5pct']['r']:.4f}")
    print(f"  Noise 10% : r={robustness_results['noise_10pct']['r']:.4f}")
    print(f"  Noise 20% : r={robustness_results['noise_20pct']['r']:.4f}")
    print(f"  Outliers  : r={robustness_results['outliers_5pct']['r']:.4f}")
    print(f"\nSaved to {RESULTS_DIR}/xgb_robustness_metrics.json")
    print(f"         {RESULTS_DIR}/xgb_cv_metrics.json")


if __name__ == "__main__":
    main()