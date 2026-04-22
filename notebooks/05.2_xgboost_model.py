"""
05_xgboost_model.py — XGBoost training, hyperparameter tuning, and SHAP analysis.
Run from project root: python notebooks/05_xgboost_model.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"pearson_r": float(r), "mae": float(mae), "rmse": float(rmse)}

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X = pd.read_csv(DATA_DIR / "X.csv")
    y = pd.read_csv(DATA_DIR / "y.csv")["popularity_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 1. XGBoost Hyperparameter Tuning (GridSearchCV)
    # ---------------------------------------------------------
    print("\nRunning 5-Fold GridSearchCV (This might take a minute or two)...")
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_lambda': [1.0, 5.0],
        'reg_alpha': [0, 1.0]
    }

    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n--- Best Hyperparameters ---")
    print(json.dumps(grid_search.best_params_, indent=2))

    # Evaluate test performance
    preds = best_model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    print(f"\n--- XGBoost Test Performance ---")
    print(f"  Test: r={metrics['pearson_r']:.4f}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")

    # ---------------------------------------------------------
    # 2. Interventional Tree SHAP Analysis
    # ---------------------------------------------------------
    print("\nCalculating Interventional Tree SHAP values...")
    # Use X_train as the background dataset for the interventional variant (TA's request)
    explainer = shap.TreeExplainer(best_model, data=X_train, feature_perturbation="interventional")
    
    # Calculate standard SHAP values on the test set
    shap_values = explainer(X_test)

    # Calculate SHAP Interaction values
    print("Calculating SHAP Interaction values...")
    interaction_values = explainer.shap_interaction_values(X_test)

    # ---------------------------------------------------------
    # 3. Save Artifacts & Plots
    # ---------------------------------------------------------
    print("Saving SHAP plots to results/ directory...")
    
    # Global Importance Plot (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_global_importance.png", dpi=300)
    plt.close()

    # Interaction Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(interaction_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_interactions.png", dpi=300)
    plt.close()

    # Save model parameters & metrics
    with open(RESULTS_DIR / "xgboost_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "best_params": grid_search.best_params_}, f, indent=2)

    print(f"\nEvaluation Complete! Artifacts saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()