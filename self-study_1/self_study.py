"Author: Iacopo Ruggero"


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import self_study_functions as funct
import importlib

 
importlib.reload(funct)
 
PATH          = Path.cwd()
output_folder = PATH / "Plots"
output_folder.mkdir(parents=True, exist_ok=True)
 
# =========================================================
# VARIABLE LISTS 
# =========================================================
 
# Variables grouped by type for inspection loops
categorical_dummy_vars = [
    "active_participation", "diploma_awarded", "disability",
    "education", "employment_gaps", "gender", "immigrant",
    "offer", "region", "takeup"
]
numeric_vars_raw    = ["outcome", "age", "special_trainings", "baseline_income",
                       "unemp_duration", "course_hours_completed"]
percentage_vars_raw = ["language_score"]
 
# Pre-treatment covariates used in all estimators
covars = [
    "age", "gender", "education", "special_trainings", "employment_gaps",
    "immigrant", "disability", "baseline_income", "unemp_duration", "language_score"
]
# Same list without region: used for within-region PS estimation (region is implicit)
covars_no_region = covars
 
# Post-treatment variables to be dropped before any analysis
post_treatment_vars = ["active_participation", "course_hours_completed", "diploma_awarded"]
 
# Binary variables that should only take values {0, 1}
binary_cols = ["offer", "takeup", "gender", "employment_gaps",
               "immigrant", "disability", "active_participation", "diploma_awarded"]
 
# Variables required to be non-missing in the complete-case sample
required_vars = ["takeup", "outcome"] + covars
 
 
# =========================================================
# QUESTION 1 — DATA LOADING AND CLEANING
# =========================================================
 
df  = pd.read_csv("data_76.csv")
raw = df.copy()  # keep a raw backup for shape comparison after cleaning
 
# ----------------------------------------------------------
# 1a. Inspect raw data — check coding, ranges, and missingness before any changes
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("RAW DATA — CATEGORICAL AND BINARY VARIABLES")
print("=" * 70)
for var in categorical_dummy_vars:
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_categorical_dummies(df, var))
 
print("\n" + "=" * 70)
print("RAW DATA — NUMERIC VARIABLES")
print("=" * 70)
for var in numeric_vars_raw:
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_numeric(df, var))
 
print("\n" + "=" * 70)
print("RAW DATA — PERCENTAGE VARIABLES")
print("=" * 70)
for var in percentage_vars_raw:
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_percentage(df, var))
 
# ----------------------------------------------------------
# 1b. Harmonize categorical labels
#     Multiple spellings observed in raw data -> map to single standard label
# ----------------------------------------------------------
region_map = {
    "A": "A", "Region A": "A", "A ": "A", "a": "A",
    "B": "B", "Region B": "B", "B-": "B", "b": "B",
    "C": "C", "Region C": "C", "c": "C", "C_": "C",
}
df["region"] = df["region"].replace(region_map)
 
education_map = {
    "0": "low",      "low": "low",       "LOW": "low",
    "1": "medium",   "med": "medium",    "Medium": "medium",
    "2": "high",     "High": "high",     "H": "high",
    "3": "tertiary", "Tertiary": "tertiary", "uni": "tertiary",
}
df["education"] = (df["education"].astype(str).str.strip()
                                  .replace(education_map))
 
# Make education an ordered categorical so level ordering is preserved in plots/tables
edu_order       = ["low", "medium", "high", "tertiary"]
df["education"] = pd.Categorical(df["education"], categories=edu_order, ordered=True)
 
# ----------------------------------------------------------
# 1c. Fix miscoded values -> recode as missing (NaN)
# ----------------------------------------------------------
df["takeup"] = df["takeup"].replace(99, np.nan)   # 99 is a placeholder for missing
 
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[(df["age"] < 18) | (df["age"] > 67),"age"] = np.nan  # implausible ages
 
df["unemp_duration"] = pd.to_numeric(df["unemp_duration"], errors="coerce")
df.loc[df["unemp_duration"] == 999, "unemp_duration"] = np.nan  # 999 is a placeholder
 
df["language_score"] = pd.to_numeric(df["language_score"], errors="coerce")
df.loc[(df["language_score"] < 0) | (df["language_score"] > 100), "language_score"] = np.nan  # outside valid range
 
# Ensure all remaining numeric columns are stored as float (not object)
numeric_cols = [
    "offer", "takeup", "outcome", "gender", "special_trainings",
    "employment_gaps", "immigrant", "disability", "baseline_income",
    "active_participation", "course_hours_completed", "diploma_awarded",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
# ----------------------------------------------------------
# 1d. Validate binary variables — any value outside {0, 1} is treated as missing
# ----------------------------------------------------------
for col in binary_cols:
    df.loc[~df[col].isin([0, 1, np.nan]), col] = np.nan


# ----------------------------------------------------------
# 1e. Drop always-takers in Region A (offer=0, takeup=1)
#     These individuals participated without receiving an offer, which is
#     inconsistent with the randomized offer mechanism in Region A.
#     They represent ~1.1% of non-offered individuals and would add noise
#     to both the selection-on-observables and IV analyses.
# ----------------------------------------------------------
always_takers_mask = (df["region"] == "A") & (df["offer"] == 0) & (df["takeup"] == 1)
print(f"\nAlways-takers dropped (Region A, offer=0, takeup=1): {always_takers_mask.sum()}")
df = df[~always_takers_mask].copy()
 
# ----------------------------------------------------------
# 1f. Drop post-treatment variables
#     Conditioning on these would introduce post-treatment bias
# ----------------------------------------------------------
df_analysis = df.drop(columns=post_treatment_vars)
 
# ----------------------------------------------------------
# 1g. Complete-case sample for causal estimation
#     Dropping rows with any missing value in required_vars ensures all estimators
#     use the exact same sample, making their results directly comparable
# ----------------------------------------------------------
df_cc = df_analysis.dropna(subset=required_vars).copy()
# Create dummies here for the saved CSV only — estimators handle dummies internally
df_cc = pd.get_dummies(df_cc, columns=["region", "education"], drop_first=True)
 
# ----------------------------------------------------------
# 1h. Cleaning summary — verify recoding worked as expected
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)
print(f"Raw shape:                          {raw.shape}")
print(f"After dropping post-treatment vars: {df_analysis.shape}")
print(f"Complete-case shape:                {df_cc.shape}")
print("\nMissing values after cleaning:")
print(df_analysis[required_vars].isna().sum())
print("\nRegion distribution:")
print(df_analysis["region"].value_counts(dropna=False))
print("\nEducation distribution:")
print(df_analysis["education"].value_counts(dropna=False))
print("\nTakeup distribution:")
print(df_analysis["takeup"].value_counts(dropna=False))
 
df_analysis.to_csv("data_76_cleaned.csv",              index=False)
df_cc.to_csv("data_76_cleaned_complete_cases.csv",     index=False)
 
# ----------------------------------------------------------
# 1i. Inspect cleaned data — confirm issues from 1a are resolved
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("CLEANED DATA — CATEGORICAL AND BINARY VARIABLES")
print("=" * 70)
for var in categorical_dummy_vars:
    if var not in df_analysis.columns:
        continue  # skip post-treatment vars already dropped
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_categorical_dummies(df_analysis, var))
 
print("\n" + "=" * 70)
print("CLEANED DATA — NUMERIC VARIABLES")
print("=" * 70)
for var in numeric_vars_raw:
    if var not in df_analysis.columns:
        continue  # skip post-treatment vars already dropped
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_numeric(df_analysis, var))
 
print("\n" + "=" * 70)
print("CLEANED DATA — PERCENTAGE VARIABLES")
print("=" * 70)
for var in percentage_vars_raw:
    print(f"\n{'-'*25} {var.upper()} {'-'*25}")
    print(funct.inspect_percentage(df_analysis, var))
 
 
# =========================================================
# QUESTION 2 — EXPLORATORY DATA ANALYSIS
# =========================================================
 
# ----------------------------------------------------------
# 2a. Region summary table — key statistics by region for comparison
# ----------------------------------------------------------
summary = pd.DataFrame({
    "n":                    df.groupby("region").size(),
    "offer_rate":           df.groupby("region")["offer"].mean(),
    "takeup_rate":          df.groupby("region")["takeup"].mean(),
    "mean_age":             df.groupby("region")["age"].mean(),
    "female_share":         df.groupby("region")["gender"].mean(),
    "mean_baseline_income": df.groupby("region")["baseline_income"].mean(),
    "mean_unemp_months":    df.groupby("region")["unemp_duration"].mean(),
    "mean_language_score":  df.groupby("region")["language_score"].mean(),
    "mean_outcome":         df.groupby("region")["outcome"].mean(),
}).round(3)
 
print("\n" + "=" * 70)
print("REGION SUMMARY TABLE")
print("=" * 70)
print(summary)
summary.to_csv("region_summary.csv")
 
# ----------------------------------------------------------
# 2b. Covariate balance by region
#     SMD between treated and control within each region
#     |SMD| < 0.1 is the conventional threshold for good balance
# ----------------------------------------------------------
balance_vars = [v for v in covars if v != "region"]  # region is a stratification var; education now handled inside balance function
 
print("\n" + "=" * 70)
print("COVARIATE BALANCE BY REGION (SMD: treated vs control)")
print("=" * 70)
for r in ["A", "B", "C"]:
    print(f"\nRegion {r}:")
    print(funct.covariate_balance_table(df_analysis, "takeup", balance_vars, region=r))
 
# ----------------------------------------------------------
# 2c. Within-region propensity score estimation
#     PS estimated separately per region WITHOUT the offer variable
#     to assess overlap based purely on pre-treatment covariates.
# ----------------------------------------------------------
ps_num = ["age", "special_trainings", "baseline_income", "unemp_duration", "language_score"]
ps_bin = ["gender", "employment_gaps", "immigrant", "disability"]
ps_cat = ["education"]
 
def fit_within_region_ps(sub, covars_list):
    # Fits a logistic regression PS model within a single region's subsample.
    # Delegates entirely to build_ps_features (via funct.estimate_propensity_scores)
    # so variable filtering is handled consistently in one place.
    # "region" is absent from covars_list here, so build_ps_features will not
    # include a region encoder — avoiding the zero-variance dummy bug.
    return funct.estimate_propensity_scores(sub, covars_list, "takeup")
 
# Compute and print overlap statistics for each region
overlap_frames = []
for r in ["A", "B", "C"]:
    sub   = df_analysis.loc[df_analysis["region"] == r,
                            covars_no_region + ["takeup"]].dropna().copy()
    ps    = fit_within_region_ps(sub, covars_no_region)
    frame = funct.ps_overlap_stats(ps, sub["takeup"].values, region_label=r)
    frame.columns = [r]
    overlap_frames.append(frame)
 
# Combine into a single comparison table across regions
overlap_combined = pd.concat(overlap_frames, axis=1)
print("\n" + "=" * 70)
print("OVERLAP STATISTICS — ALL REGIONS")
print("=" * 70)
print(overlap_combined.to_string())
overlap_combined.to_csv(PATH / "ps_overlap_stats.csv")
 
# ----------------------------------------------------------
# 2d. Plots
# ----------------------------------------------------------
 
# Plot 1: offer and takeup rates by region — highlights Region A's distinct mechanism
rates = summary[["offer_rate", "takeup_rate"]]
ax    = rates.plot(kind="bar", figsize=(8, 5))
ax.set_title("Offer and takeup rates by region")
ax.set_ylabel("Rate")
ax.set_xlabel("Region")
plt.tight_layout()
plt.savefig(output_folder / "offer_takeup_by_region.png", dpi=180)
plt.close()
 
# Plot 2: education composition by region — stacked bar for compositional comparison
edu = pd.crosstab(df["region"], df["education"],
                  normalize="index")[["low", "medium", "high", "tertiary"]]
ax  = edu.plot(kind="bar", stacked=True, figsize=(8, 5))
ax.set_title("Education composition by region")
ax.set_ylabel("Share")
ax.set_xlabel("Region")
plt.tight_layout()
plt.savefig(output_folder / "education_by_region.png", dpi=180)
plt.close()
 
# Plot 3: age distribution by region — density histogram to spot Region C's older population
plt.figure(figsize=(8, 5))
for r in ["A", "B", "C"]:
    vals = df.loc[df["region"] == r, "age"].dropna()
    plt.hist(vals, bins=30, alpha=0.5, density=True, label=r)
plt.legend(title="Region")
plt.title("Age distribution by region")
plt.xlabel("Age")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(output_folder / "age_by_region.png", dpi=180)
plt.close()
 
# Plot 4: Region A takeup by offer — confirms offer drives take-up (IV relevance)
tab = pd.crosstab(df.loc[df["region"] == "A", "offer"],
                  df.loc[df["region"] == "A", "takeup"],
                  normalize="index")
tab.index   = ["No offer", "Offer"]
tab.columns = ["No takeup", "Takeup"]
ax = tab[["No takeup", "Takeup"]].plot(kind="bar", stacked=True, figsize=(7, 5))
ax.set_title("Region A: takeup by offer status")
ax.set_ylabel("Share")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(output_folder / "regionA_takeup_by_offer.png", dpi=180)
plt.close()
 
# Plot 5: PS overlap within each region (estimated without offer)
# Side-by-side panels show treated vs control PS distributions per region
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, r in zip(axes, ["A", "B", "C"]):
    sub      = df_analysis.loc[df_analysis["region"] == r,
                               covars_no_region + ["takeup"]].dropna().copy()
    ps       = fit_within_region_ps(sub, covars_no_region)
    sub["ps"] = ps
    ax.hist(sub.loc[sub["takeup"] == 0, "ps"], bins=25, alpha=0.6, density=True, label="Control")
    ax.hist(sub.loc[sub["takeup"] == 1, "ps"], bins=25, alpha=0.6, density=True, label="Treated")
    ax.set_title(f"Region {r}")
    ax.set_xlabel("Estimated propensity score")
axes[0].set_ylabel("Density")
axes[0].legend()
fig.suptitle("Overlap within regions (propensity scores estimated without offer)")
fig.tight_layout()
plt.savefig(output_folder / "ps_overlap_by_region.png", dpi=180)
plt.close()
 
# Plot 6: baseline income by region — boxplot to visualize Region C's higher income
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column="baseline_income", by="region", ax=ax)
ax.set_title("Baseline income by region")
ax.set_xlabel("Region")
ax.set_ylabel("Baseline income")
plt.suptitle("")  # suppress automatic pandas figure-level title
fig.savefig(output_folder / "baseline_income_by_region.png", dpi=180)
plt.close()
 
# Per-variable plots: binary (bar by region) and numeric (histogram + boxplot by region)
binary_plot_vars  = ["offer", "takeup", "gender", "employment_gaps", "immigrant", "disability"]
numeric_plot_vars = ["outcome", "age", "special_trainings",
                     "baseline_income", "unemp_duration", "language_score"]
 
for var in binary_plot_vars:
    if var in df_analysis.columns:
        funct.plot_binary_by_region(df_analysis, var, output_folder)
 
for var in numeric_plot_vars:
    if var in df_analysis.columns:
        funct.plot_numeric_hist_by_region(df_analysis, var, output_folder)
        funct.plot_numeric_box_by_region(df_analysis,  var, output_folder)
 
# Covariate balance dot plot — saved for Q4 figure in report
funct.plot_covariate_balance(df_analysis, "takeup", covars, output_folder)
 
print(f"\nAll Q2 plots saved to: {output_folder}")
 
 
# =========================================================
# QUESTION 3 — ESTIMATION UNDER SELECTION ON OBSERVABLES
# =========================================================
 
# Filter to Regions A and B: best-behaved overlap, most suitable for selection-on-observables
df_AB = df_analysis[df_analysis["region"].isin(["A", "B"])].copy()
 
y_var  = "outcome"
d_var  = "takeup"
x_vars = covars + ["region"]  # region included as covariate to control for regional differences
 
# Drop any remaining missing values — complete-case sample for estimation
estimation_vars = [y_var, d_var] + x_vars
df_est          = df_AB[estimation_vars].dropna().copy()
 
Y = df_est[y_var].to_numpy(dtype=float)
D = df_est[d_var].to_numpy(dtype=float)
 
print("\n" + "=" * 70)
print("QUESTION 3 — TREATMENT EFFECT ESTIMATION (Regions A & B)")
print("=" * 70)
print(f"Observations: {len(df_est)}  |  Regions: {sorted(df_est['region'].unique())}")
 
# Run all five estimators and collect results
estimator_results = []
for label, res in [
    ("Raw mean difference",   funct.raw_mean_difference(Y, D)),
    ("Regression adjustment", funct.regression_adjustment(df_est, y_var, d_var, x_vars)),
    ("IPW",                   funct.ipw_ate(df_est, y_var, d_var, x_vars)),
    ("Doubly robust",         funct.doubly_robust_ate(df_est, y_var, d_var, x_vars)),
]:
    estimator_results.append([label, res["estimate"], res["se"], res["z"], res["p_value"]])

# Normalized IPW with bootstrap SE — estimated separately, then inserted into table
ipw_norm_res = funct.ipw_ate_normalized_bootstrap(
    df_est, y_var, d_var, x_vars,
    n_bootstrap=500,
    seed=42
)
estimator_results.insert(
    3,  # place after standard IPW, before Doubly robust
    ["IPW (normalized, bootstrap SE)",
     ipw_norm_res["estimate"], ipw_norm_res["se"],
     ipw_norm_res["z"],        ipw_norm_res["p_value"]]
)

results_table = pd.DataFrame(
    estimator_results,
    columns=["method", "estimate", "se", "z_stat", "p_value"]
).round({"estimate": 3, "se": 3, "z_stat": 3, "p_value": 4})
 
results_table["signif"] = results_table["p_value"].apply(funct.stars)  # add significance stars
 
print("\n", results_table.to_string(index=False))
results_table.to_csv(output_folder / "treatment_effect_estimates_AB.csv", index=False)
print(f"\nSaved to: {output_folder / 'treatment_effect_estimates_AB.csv'}")