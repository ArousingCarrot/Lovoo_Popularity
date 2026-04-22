"""
04_baseline_lr.py — OLS baseline for Lovoo popularity prediction.
Run from project root: python notebooks/04_baseline_lr.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(DATA_DIR / "X.csv")
    y_df = pd.read_csv(DATA_DIR / "y.csv")
    validate_target_column(y_df)
    y = y_df["popularity_score"]
    return X, y


def validate_target_column(y_df: pd.DataFrame) -> None:
    if "popularity_score" not in y_df.columns:
        available = ", ".join(str(c) for c in y_df.columns)
        raise ValueError(
            f"'popularity_score' not found in y.csv. "
            f"Available columns: {available}"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"pearson_r": float(r), "mae": float(mae), "rmse": float(rmse)}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    fold_r, fold_rmse = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr), start=1):
        model = LinearRegression()
        model.fit(X_arr[train_idx], y_arr[train_idx])
        preds = model.predict(X_arr[val_idx])
        m = compute_metrics(y_arr[val_idx], preds)
        fold_r.append(m["pearson_r"])
        fold_rmse.append(m["rmse"])
        print(f"  Fold {fold_idx}: r={m['pearson_r']:.4f}  RMSE={m['rmse']:.4f}")

    return {
        "cv_pearson_r_folds": fold_r,
        "cv_rmse_folds": fold_rmse,
        "cv_pearson_r_mean": float(np.mean(fold_r)),
        "cv_pearson_r_std": float(np.std(fold_r, ddof=1)),
        "cv_rmse_mean": float(np.mean(fold_rmse)),
        "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
    }


# ---------------------------------------------------------------------------
# Artifact savers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_coefficients(feature_names: list[str], coef: np.ndarray, path: Path) -> None:
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    df.to_csv(path, index=False)


def save_predictions(
    y_train: np.ndarray, pred_train: np.ndarray,
    y_test: np.ndarray, pred_test: np.ndarray,
    path: Path,
) -> None:
    train_df = pd.DataFrame({
        "split": "train",
        "y_true": y_train,
        "y_pred": pred_train,
        "residual": y_train - pred_train,
    })
    test_df = pd.DataFrame({
        "split": "test",
        "y_true": y_test,
        "y_pred": pred_test,
        "residual": y_test - pred_test,
    })
    pd.concat([train_df, test_df], ignore_index=True).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    n_samples, n_features = X.shape
    print(f"  X: {n_samples} rows × {n_features} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("\nFitting OLS baseline...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    test_metrics = compute_metrics(y_test.to_numpy(), pred_test)
    print(f"  Test — r={test_metrics['pearson_r']:.4f}  "
          f"MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")

    print(f"\nRunning {N_FOLDS}-fold cross-validation (full dataset)...")
    cv_results = cross_validate_model(X, y)

    all_metrics = {
        "n_samples": n_samples,
        "n_features": n_features,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "test_pearson_r": test_metrics["pearson_r"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        **cv_results,
    }

    save_metrics(all_metrics, RESULTS_DIR / "baseline_lr_metrics.json")
    save_coefficients(X.columns.tolist(), model.coef_, RESULTS_DIR / "baseline_lr_coefficients.csv")
    save_predictions(
        y_train.to_numpy(), pred_train,
        y_test.to_numpy(), pred_test,
        RESULTS_DIR / "baseline_lr_predictions.csv",
    )

    print("\n--- Baseline LR Summary ---")
    print(f"  Samples : {n_samples}  |  Features: {n_features}")
    print(f"  Test    : r={test_metrics['pearson_r']:.4f}  "
          f"MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}")
    print(f"  CV {N_FOLDS}-fold : r={cv_results['cv_pearson_r_mean']:.4f} "
          f"± {cv_results['cv_pearson_r_std']:.4f}  "
          f"RMSE={cv_results['cv_rmse_mean']:.4f} "
          f"± {cv_results['cv_rmse_std']:.4f}")
    print(f"\nArtifacts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
