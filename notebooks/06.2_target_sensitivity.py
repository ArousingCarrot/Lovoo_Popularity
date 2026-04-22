"""
06_target_sensitivity.py — Robustness check for target variable construction.
Run from project root: python notebooks/06_target_sensitivity.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

def main():
    print("Loading data...")
    X = pd.read_csv(DATA_DIR / "X.csv")
    y_pc1 = pd.read_csv(DATA_DIR / "y.csv")["popularity_score"]
    
    # Load the raw engagement metrics
    y_raw = pd.read_csv(DATA_DIR / "y_raw.csv")
    
    # The four metrics mentioned in your proposal
    raw_metrics = ["counts_profileVisits", "counts_kisses", "counts_fans", "counts_gifts"]
    
    # Verify columns exist, adjust if naming is slightly different in your csv
    available_cols = y_raw.columns.tolist()
    metrics_to_use = [col for col in raw_metrics if col in available_cols]
    
    if not metrics_to_use:
        print(f"Error: Could not find the raw metric columns in y_raw.csv. Available columns: {available_cols}")
        return

    # 1. Create Target A: Equal-weight sum of log-transformed metrics
    y_raw['equal_weight_target'] = 0
    for col in metrics_to_use:
        y_raw[f'log_{col}'] = np.log1p(y_raw[col]) # log(1+x) transformation
        y_raw['equal_weight_target'] += y_raw[f'log_{col}']

    # 2. Define all our targets for comparison
    targets = {
        "1. PC1 (Original Baseline)": y_pc1,
        "2. Equal-Weight Sum": y_raw['equal_weight_target']
    }
    
    # Add Target B: Each raw metric individually
    for col in metrics_to_use:
        targets[f"3. Individual: {col}"] = y_raw[f'log_{col}']

    # 3. Evaluate Feature Importance Stability
    print("\n" + "="*50)
    print(" TARGET SENSITIVITY CHECK: TOP 5 FEATURES")
    print("="*50)
    
    for target_name, target_series in targets.items():
        model = LinearRegression()
        model.fit(X, target_series)

        # Extract and sort coefficients
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "abs_coef": np.abs(model.coef_)
        }).sort_values(by="abs_coef", ascending=False).head(5).reset_index(drop=True)

        print(f"\n{target_name}")
        print("-" * 30)
        for idx, row in coef_df.iterrows():
            print(f"  {idx + 1}. {row['feature']:<20} ({row['abs_coef']:.4f})")

if __name__ == "__main__":
    main()