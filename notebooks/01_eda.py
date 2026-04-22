import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/raw"
FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
api = pd.read_csv(f"{DATA_DIR}/lovoo_v3_users_api-results.csv")
instances = pd.read_csv(f"{DATA_DIR}/lovoo_v3_users_instances.csv")
interests = pd.read_csv(f"{DATA_DIR}/interests-of-users-by-age.csv", index_col="index")

print(f"API shape: {api.shape}")
print(f"Instances shape: {instances.shape}")

# ── 2. Dedup and merge ────────────────────────────────────────────────────────
def dedup(df, label):
    n = len(df)
    if "lastOnlineDate" in df.columns:
        df = df.sort_values("lastOnlineDate", ascending=False)
    df = df.drop_duplicates(subset="userId", keep="first")
    print(f"{label}: {n} -> {len(df)} after dedup")
    return df

api = dedup(api, "API")
instances = dedup(instances, "Instances")

overlap = set(api["userId"]) & set(instances["userId"])
print(f"Overlapping userIds: {len(overlap)}")

df = api.merge(instances, on="userId", how="left", suffixes=("", "_inst"))
dup_cols = [c for c in df.columns if c.endswith("_inst")]
df.drop(columns=dup_cols, inplace=True)
print(f"Merged shape: {df.shape}")

# ── 3. Missingness ────────────────────────────────────────────────────────────
miss = df.isnull().mean().sort_values(ascending=False)
miss = miss[miss > 0]
print("\nMissing value rates:\n", miss.to_string())

fig, ax = plt.subplots(figsize=(8, max(4, len(miss) * 0.35)))
miss.plot.barh(ax=ax)
ax.set_xlabel("Missing fraction")
ax.set_title("Missing Values by Feature")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/missingness.png", dpi=150)
plt.show()

# ── 4. Engagement metrics ─────────────────────────────────────────────────────
ENG_COLS = ["counts_profileVisits", "counts_kisses", "counts_fans", "counts_g"]
print("\n--- Engagement summary (raw) ---")
print(df[ENG_COLS].describe())

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, ENG_COLS):
    df[col].dropna().clip(upper=df[col].quantile(0.99)).hist(bins=50, ax=ax)
    ax.set_title(col)
plt.suptitle("Engagement Distributions (clipped at 99th pct)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/engagement_raw.png", dpi=150)
plt.show()

log_eng = df[ENG_COLS].apply(np.log1p)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, ENG_COLS):
    log_eng[col].dropna().hist(bins=50, ax=ax)
    ax.set_title(f"log1p({col})")
plt.suptitle("Engagement Distributions (log1p)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/engagement_log.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(log_eng.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Engagement Correlation (log1p)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/engagement_corr.png", dpi=150)
plt.show()

# ── 5. Demographics ───────────────────────────────────────────────────────────
print("\nGender value counts:\n", df["gender"].value_counts())

fig, ax = plt.subplots(figsize=(6, 3))
df["age"].dropna().clip(upper=80).hist(bins=40, ax=ax)
ax.set_title("Age Distribution")
ax.set_xlabel("age")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/age_dist.png", dpi=150)
plt.show()

print(f"\nUnique countries: {df['country'].nunique()}")
print(df["country"].value_counts().head(15))

# ── 6. Text fields ────────────────────────────────────────────────────────────
for col in ["freetext", "whazzup"]:
    if col in df.columns:
        df[f"{col}_len"] = df[col].fillna("").str.len()
        miss_rate = (df[col].isnull() | (df[col] == "")).mean()
        print(f"\n{col}: {miss_rate:.1%} empty/missing")
        print(df[f"{col}_len"].describe())

# ── 7. Self-presentation features ────────────────────────────────────────────
SELFPRES = ["counts_pictures", "counts_details", "freetext_len", "whazzup_len"]
selfpres_present = [c for c in SELFPRES if c in df.columns]
print("\n--- Self-presentation summary ---")
print(df[selfpres_present].describe())

# ── 8. Platform flags and intent ─────────────────────────────────────────────
CONFOUND = ["isVip", "isHighlighted", "isInfluencer", "isNew", "isFlirtstar",
            "isMobile", "isOnline", "isSystemProfile", "verified"]
INTENT   = ["flirtInterests_chat", "flirtInterests_friends", "flirtInterests_date"]
flag_cols = [c for c in CONFOUND + INTENT if c in df.columns]
flag_rates = (
    df[flag_cols]
    .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0))
    .mean()
    .sort_values(ascending=False)
)
print("\nFlag/intent rates:\n", flag_rates.to_string())

# ── 9. Language features ──────────────────────────────────────────────────────
LANG_COLS = ["lang_fr", "lang_en", "lang_de", "lang_it", "lang_es", "lang_pt"]
lang_rates = df[LANG_COLS].apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0)).mean()
print("\nLanguage rates:\n", lang_rates.to_string())
print(f"\nlang_count:\n{df['lang_count'].describe()}")

# ── 10. Feature vs engagement correlation ─────────────────────────────────────
NUM_FEATURES = ["age", "counts_pictures", "counts_details",
                "freetext_len", "whazzup_len", "lang_count"]
num_present = [c for c in NUM_FEATURES if c in df.columns]

corr_block = pd.concat([df[num_present], log_eng], axis=1).corr().loc[num_present, ENG_COLS]
print("\n--- Feature vs Engagement Correlation ---")
print(corr_block.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(corr_block, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Feature vs Engagement Correlation (log1p engagement)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_engagement_corr.png", dpi=150)
plt.show()

# ── 11. Interests-by-age ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["meankissesreceived", "meanvisitsreceived", "meanfollowers"]):
    ax.plot(interests["age"], interests[col], marker="o", markersize=4)
    ax.set_xlabel("age")
    ax.set_title(col)
plt.suptitle("Mean Engagement by Age Group")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/engagement_by_age.png", dpi=150)
plt.show()

# ── 12. Save ──────────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/lovoo_deduped.csv", index=False)
print(f"\nSaved to data/processed/lovoo_deduped.csv")
print(f"Final shape: {df.shape}")
print(f"EDA complete. Figures saved to {FIG_DIR}/")