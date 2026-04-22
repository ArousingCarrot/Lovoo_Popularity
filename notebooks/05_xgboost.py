"""
05_xgboost_shap.py — XGBoost + SHAP analysis
Run: python notebooks/05_xgboost_shap.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    X = pd.read_csv(DATA_DIR / "X.csv")
    y = pd.read_csv(DATA_DIR / "y.csv")["popularity_score"]
    return X, y

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"pearson_r": float(r), "mae": float(mae), "rmse": float(rmse)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data()

    print("\nFeatures used:")
    print(X.columns.tolist())

    # -----------------------------------------------------------------------
    # Train/test split
    # -----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # -----------------------------------------------------------------------
    # Grid search (TRAIN ONLY)
    # -----------------------------------------------------------------------

    print("\nRunning Grid Search...")

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 5],
        "reg_alpha": [0, 1],
    }

    model = XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1
    )

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest params:")
    print(grid.best_params_)

    # -----------------------------------------------------------------------
    # Save model + metadata
    # -----------------------------------------------------------------------

    joblib.dump(best_model, RESULTS_DIR / "xgb_model.joblib")

    with open(RESULTS_DIR / "feature_names.json", "w") as f:
        json.dump(X.columns.tolist(), f)

    with open(RESULTS_DIR / "xgb_best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    # -----------------------------------------------------------------------
    # Predictions (THIS IS WHAT YOU ASKED FOR)
    # -----------------------------------------------------------------------

    pred_test = best_model.predict(X_test)

    pred_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": pred_test,
        "residual": y_test.values - pred_test,
        "abs_error": np.abs(y_test.values - pred_test)
    })

    pred_df.to_csv(RESULTS_DIR / "xgb_predictions.csv", index=False)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    test_metrics = compute_metrics(y_test.values, pred_test)

    print(f"\nTest — r={test_metrics['pearson_r']:.4f} "
          f"MAE={test_metrics['mae']:.4f} "
          f"RMSE={test_metrics['rmse']:.4f}")

    # -----------------------------------------------------------------------
    # XGBoost gain importance
    # -----------------------------------------------------------------------

    gain_df = pd.DataFrame({
        "feature": X.columns,
        "gain_importance": best_model.feature_importances_
    }).sort_values("gain_importance", ascending=False)

    gain_df.to_csv(RESULTS_DIR / "xgb_gain_importance.csv", index=False)

    # -----------------------------------------------------------------------
    # SHAP (Tree SHAP, interventional)
    # -----------------------------------------------------------------------

    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(
        best_model,
        feature_perturbation="interventional"
    )

    shap_values = explainer(X_test)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    # -----------------------------------------------------------------------
    # SHAP interactions
    # -----------------------------------------------------------------------

    print("Computing SHAP interaction values...")

    shap_interaction = explainer.shap_interaction_values(X_test)

    interaction_matrix = np.abs(shap_interaction).mean(axis=0)

    interaction_df = pd.DataFrame(
        interaction_matrix,
        index=X.columns,
        columns=X.columns
    )

    interaction_df.to_csv(RESULTS_DIR / "shap_interactions_matrix.csv")

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    all_metrics = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "test_pearson_r": test_metrics["pearson_r"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "best_params": grid.best_params_,
    }

    with open(RESULTS_DIR / "xgb_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------

    print("\n--- XGBoost + SHAP Summary ---")
    print(f"Samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Test r={test_metrics['pearson_r']:.4f} "
          f"RMSE={test_metrics['rmse']:.4f}")
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()