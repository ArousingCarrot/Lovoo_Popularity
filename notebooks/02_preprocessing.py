import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os

DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/lovoo_deduped.csv")
print(f"Loaded: {df.shape}")

# ── 2. Drop near-zero signal columns ─────────────────────────────────────────
DROP_COLS = ["crypt", "locationCitySub", "userInfo_visitDate", "isSystemProfile",
             "freetext", "freetext_len", "name", "pictureId", "location",
             "locationCity", "city", "distance"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
print(f"Shape after dropping sparse/irrelevant columns: {df.shape}")

# ── 3. Impute structural missingness (27% block from instances) ───────────────
INSTANCE_BINARY = ["online", "mobile", "locked", "connectedToFacebook",
                   "highlighted", "hasBirthday", "freshman", "flirtstar",
                   "isVerified", "isVIP"]
for col in INSTANCE_BINARY:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# countDetails from instances: fill with 0
if "countDetails" in df.columns:
    df["countDetails"] = pd.to_numeric(df["countDetails"], errors="coerce").fillna(0)

# whazzup: missing = no text
df["whazzup"] = df["whazzup"].fillna("")

# Remaining numerics: fill with median
MEDIAN_COLS = ["age", "counts_pictures", "counts_details", "lang_count",
               "counts_profileVisits", "counts_kisses", "counts_fans", "counts_g"]
for col in MEDIAN_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

print(f"Missing values remaining: {df.isnull().sum().sum()}")

# ── 4. Save raw engagement for PCA (next step) ────────────────────────────────
ENG_COLS = ["counts_profileVisits", "counts_kisses", "counts_fans", "counts_g"]
y_raw = pd.DataFrame(np.log1p(df[ENG_COLS].astype(float)), columns=ENG_COLS)
y_raw.to_csv(f"{DATA_DIR}/y_raw.csv", index=False)
print(f"Saved y_raw: {y_raw.shape}")

# ── 5. Feature engineering ────────────────────────────────────────────────────

# Country -> binary EU indicator (31 unique values, too sparse for one-hot)
EU_COUNTRIES = {"DE", "CH", "FR", "IT", "AT", "BE", "ES", "LU", "BA",
                "RO", "PT", "NL", "PL", "HU", "CZ", "SK", "HR", "BG",
                "SI", "EE", "LV", "LT", "FI", "SE", "DK", "GR", "CY",
                "MT", "IE", "SC", "GB"}
df["is_EU"] = df["country"].isin(EU_COUNTRIES).astype(int)

# whazzup length as effort signal
df["whazzup_len"] = df["whazzup"].str.len()

# Gender: drop (all-female, zero variance)
df.drop(columns=["gender"], inplace=True, errors="ignore")

# ── 6. Encode categoricals ────────────────────────────────────────────────────
BINARY_COLS = (
    ["isVip", "isHighlighted", "isInfluencer", "isNew", "isFlirtstar",
     "isMobile", "isOnline", "verified", "shareProfileEnabled",
     "flirtInterests_chat", "flirtInterests_friends", "flirtInterests_date"] +
    ["lang_fr", "lang_en", "lang_de", "lang_it", "lang_es", "lang_pt"]
)
for col in BINARY_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# ── 7. Assemble feature matrix ────────────────────────────────────────────────
DEMO = ["age", "is_EU"]
SELFPRES = ["counts_pictures", "counts_details", "whazzup_len"]
INTENT = ["flirtInterests_chat", "flirtInterests_friends", "flirtInterests_date"]
LANG = ["lang_count", "lang_fr", "lang_en", "lang_de", "lang_it", "lang_es", "lang_pt"]
CONFOUND = ["isVip", "isHighlighted", "isFlirtstar", "isNew",
            "isMobile", "isOnline", "isInfluencer"]

all_feature_cols = DEMO + SELFPRES + INTENT + LANG + CONFOUND
all_feature_cols = [c for c in all_feature_cols if c in df.columns]

X = df[all_feature_cols].copy()
print(f"\nFeature matrix shape: {X.shape}")
print(f"Groups: demo={len([c for c in DEMO if c in X.columns])}, "
      f"selfpres={len([c for c in SELFPRES if c in X.columns])}, "
      f"intent={len([c for c in INTENT if c in X.columns])}, "
      f"lang={len([c for c in LANG if c in X.columns])}, "
      f"confound={len([c for c in CONFOUND if c in X.columns])}")

# ── 8. Scale continuous features ─────────────────────────────────────────────
SCALE_COLS = ["age", "counts_pictures", "counts_details", "whazzup_len", "lang_count"]
scale_present = [c for c in SCALE_COLS if c in X.columns]
scaler = StandardScaler()
X[scale_present] = scaler.fit_transform(X[scale_present])

# ── 9. Save ───────────────────────────────────────────────────────────────────
X.to_csv(f"{DATA_DIR}/X.csv", index=False)
print(f"Saved X to {DATA_DIR}/X.csv")

col_groups = {
    "demo": [c for c in DEMO if c in X.columns],
    "selfpres": [c for c in SELFPRES if c in X.columns],
    "intent": [c for c in INTENT if c in X.columns],
    "lang": [c for c in LANG if c in X.columns],
    "confound": [c for c in CONFOUND if c in X.columns],
}
with open(f"{DATA_DIR}/col_groups.json", "w") as f:
    json.dump(col_groups, f, indent=2)

print("Saved col_groups.json")
print("\nPreprocessing complete.")