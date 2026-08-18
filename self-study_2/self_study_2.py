"""
Self Study Part II - Instrumental Variables
Author: Iacopo Ruggero
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

import part1_functions as func

############## OUTPUT FOLDER
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

############## 0.  PART I CLEANING
df_raw = pd.read_csv("./datasets/data_76.csv")

post_treatment = ["active_participation", "course_hours_completed", "diploma_awarded"]
df = df_raw.drop(columns=post_treatment)

df["takeup"] = df["takeup"].replace(99, np.nan)
df = df.dropna(subset=["takeup"])
df.loc[df["unemp_duration"] > 900, "unemp_duration"] = np.nan
df.loc[df["language_score"] > 100, "language_score"] = np.nan
df.loc[df["language_score"] < 0,   "language_score"] = np.nan
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df.loc[(df["age"] < 18) | (df["age"] > 70), "age"] = np.nan
df["region"] = df["region"].astype(str).str.strip().str.upper()
df["region"] = df["region"].str.extract(r"(A|B|C)", expand=False)
df.loc[df["baseline_income"] < 0,       "baseline_income"] = np.nan
df.loc[df["baseline_income"] > 200_000, "baseline_income"] = np.nan
df["education"] = df["education"].astype(str).str.strip().str.lower()
edu_map = {
    "0":"low","low":"low","1":"medium","med":"medium","medium":"medium",
    "2":"high","h":"high","high":"high","3":"tertiary","tertiary":"tertiary","uni":"tertiary"
}
df["education"] = df["education"].map(edu_map)
edu_dummies    = pd.get_dummies(df["education"], prefix="edu",    drop_first=True)
region_dummies = pd.get_dummies(df["region"],    prefix="region", drop_first=True)
df = pd.concat([df, edu_dummies, region_dummies], axis=1)
df = df.dropna()

# Keep only Region A
df_A = df[df["region"] == "A"].copy().reset_index(drop=True)
print(f"Region A observations: {len(df_A)}")

covariates = ["age", "gender", "special_trainings", "employment_gaps",
              "baseline_income", "unemp_duration", "language_score",
              "immigrant", "disability"] + list(edu_dummies.columns)

# Common arrays
n  = len(df_A)
Y  = df_A["outcome"].values.astype(float)
D  = df_A["takeup"].values.astype(float)
Z  = df_A["offer"].values.astype(float)
W  = df_A[covariates].astype(float).values


############## EXERCISE 6 - Instrument Validation
print("\n" + "#"*70)
print("# EXERCISE 6 - Instrument Validation")
print("#"*70)

############## 6a - Random assignment of offer
print("\n--- 6a: balance check ---")
offer_0 = df_A[df_A["offer"] == 0]
offer_1 = df_A[df_A["offer"] == 1]

balance_rows = []
for col in covariates:
    a = offer_1[col].dropna().values
    b = offer_0[col].dropna().values
    m1, m0, diff, t, p = func.welch_t_scratch(a, b)
    pooled_sd = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    std_diff  = diff / pooled_sd if pooled_sd > 0 else 0.0
    balance_rows.append({"Variable": col, "Mean(Z=0)": round(m0, 3),
                         "Mean(Z=1)": round(m1, 3), "Std.Diff": round(std_diff, 3),
                         "t": round(t, 3), "p": round(p, 3)})

balance_df = pd.DataFrame(balance_rows)
print(balance_df.to_string(index=False))

F_joint, p_joint = func.joint_f_scratch(W, Z)
print(f"\nJoint F (offer ~ covariates): F={F_joint:.3f}, p={p_joint:.3f}")
print(f"Offer rate: {df_A['offer'].mean():.3f}")


############## 6b - First stage: offer -> take-up
print("\n--- 6b: first stage ---")

# Without covariates
X_s = np.column_stack([np.ones(n), Z])
beta_s, resid_s, _ = func.ols_scratch(X_s, D)
se_s = func.ols_se_scratch(X_s, resid_s)
t_s  = beta_s / se_s
p_s  = 2 * (1 - scipy_stats.t.cdf(np.abs(t_s), n - X_s.shape[1]))

# With covariates
X_f = np.column_stack([np.ones(n), Z, W])
beta_f, resid_f, _ = func.ols_scratch(X_f, D)
se_f = func.ols_se_scratch(X_f, resid_f)
t_f  = beta_f / se_f
p_f  = 2 * (1 - scipy_stats.t.cdf(np.abs(t_f), n - X_f.shape[1]))

print(f"First stage (no covariates): offer coeff={beta_s[1]:.4f}, SE={se_s[1]:.4f}, t={t_s[1]:.2f}")
print(f"First stage (w/ covariates): offer coeff={beta_f[1]:.4f}, SE={se_f[1]:.4f}, t={t_f[1]:.2f}")
print(f"Compliance rate: {beta_s[1]:.4f}")


############## 6c - Instrument strength (partial F)
print("\n--- 6c: instrument strength ---")
X_r = np.column_stack([np.ones(n), W])
F_partial, p_partial = func.partial_f_scratch(X_f, X_r, D)
print(f"Partial F (offer in first stage): {F_partial:.2f}, p={p_partial:.6f}")


############## Plots for Exercise 6
sns.set_style("whitegrid")

# Balance chart (standardised differences)
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["tab:red" if p < 0.05 else "tab:green" for p in balance_df["p"]]
ax.barh(balance_df["Variable"], balance_df["Std.Diff"], color=colors)
ax.axvline(0,    color="black", linewidth=0.8)
ax.axvline(-0.1, color="grey",  linewidth=0.8, linestyle="--")
ax.axvline( 0.1, color="grey",  linewidth=0.8, linestyle="--")
ax.set_xlabel("Standardised mean difference (Z=1 minus Z=0)")
ax.set_title("Covariate balance by offer status (red = p < 0.05)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ex6_balance.png"), dpi=150)
plt.close()

# First-stage bar chart
takeup_rates = [offer_0["takeup"].mean(), offer_1["takeup"].mean()]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Not offered (Z=0)", "Offered (Z=1)"], takeup_rates,
       color=["tab:blue", "tab:orange"])
ax.set_ylabel("Take-up rate")
ax.set_title("First stage: take-up by offer")
ax.set_ylim(0, 1)
for i, v in enumerate(takeup_rates):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ex6_firststage_bar.png"), dpi=150)
plt.close()

############## EXERCISE 7 - Naive OLS for comparison
############## EXERCISE 7 - Naive OLS for comparison
print("\n" + "#"*70)
print("# EXERCISE 7 - Naive OLS (Y on D) in Region A")
print("#"*70)

# OLS without covariates
X_ols = np.column_stack([np.ones(n), D])
beta_ols, resid_ols, _ = func.ols_scratch(X_ols, Y)
se_ols   = func.ols_se_scratch(X_ols, resid_ols)
t_ols    = beta_ols[1] / se_ols[1]
p_ols    = 2 * (1 - scipy_stats.t.cdf(abs(t_ols), n - 2))
print(f"  OLS Y on D (no covariates) : {beta_ols[1]:.2f} (SE={se_ols[1]:.2f}, t={t_ols:.2f}, p={p_ols:.4f})")

# OLS with covariates
X_ols_cov = np.column_stack([np.ones(n), D, W])
beta_ols_cov, resid_ols_cov, _ = func.ols_scratch(X_ols_cov, Y)
se_ols_cov   = func.ols_se_scratch(X_ols_cov, resid_ols_cov)
t_ols_cov    = beta_ols_cov[1] / se_ols_cov[1]
p_ols_cov    = 2 * (1 - scipy_stats.t.cdf(abs(t_ols_cov), n - X_ols_cov.shape[1]))
print(f"  OLS Y on D (w/ covariates) : {beta_ols_cov[1]:.2f} (SE={se_ols_cov[1]:.2f}, t={t_ols_cov:.2f}, p={p_ols_cov:.4f})")
print("  Compare to LATE (2,618) below: OLS is biased upward by selection on unobservables.")
print("  Adding covariates barely closes the gap, consistent with selection on unobservables.")

############## EXERCISE 8 - IV Estimation
print("\n" + "#"*70)
print("# EXERCISE 8 - IV Estimation")
print("#"*70)

############## 8a - ITT: OLS of outcome on offer
print("\n--- 8a: ITT (outcome ~ offer) ---")
X_itt = np.column_stack([np.ones(n), Z])
beta_itt, resid_itt, _ = func.ols_scratch(X_itt, Y)
se_itt   = func.ols_se_scratch(X_itt, resid_itt)
t_itt    = beta_itt[1] / se_itt[1]
p_itt    = 2 * (1 - scipy_stats.t.cdf(abs(t_itt), n - 2))

print(f"  Intercept : {beta_itt[0]:.2f} (SE={se_itt[0]:.2f})")
print(f"  Offer     : {beta_itt[1]:.2f} (SE={se_itt[1]:.2f}, t={t_itt:.2f}, p={p_itt:.4f})")


############## 8b - LATE via Wald estimator
print("\n--- 8b: LATE (Wald estimator) ---")
late_wald, itt_y, itt_d = func.wald_late_scratch(Y, D, Z)
late_se  = func.wald_late_boot_scratch(Y, D, Z, reps=500)
late_t   = late_wald / late_se
late_p   = 2 * (1 - scipy_stats.norm.cdf(abs(late_t)))

print(f"  Reduced form (ITT_Y) : {itt_y:.2f}")
print(f"  First stage  (ITT_D) : {itt_d:.4f}")
print(f"  Wald LATE            : {late_wald:.2f}")
print(f"  SE (bootstrap, B=500): {late_se:.2f}")
print(f"  t-stat               : {late_t:.2f}, p-value: {late_p:.4f}")


############## 8c - 2SLS with covariates
print("\n--- 8c: 2SLS with covariates ---")
beta_2sls, se_2sls = func.two_sls_scratch(Y, D, Z, W)
coef_D, se_D       = beta_2sls[1], se_2sls[1]
t_D                = coef_D / se_D
p_D                = 2 * (1 - scipy_stats.t.cdf(abs(t_D), n - len(beta_2sls)))

print(f"  Coefficient on takeup : {coef_D:.2f}")
print(f"  Standard error        : {se_D:.2f}")
print(f"  t-statistic           : {t_D:.2f}, p-value: {p_D:.4f}")

# Sanity check: 2SLS without covariates equals the Wald estimator
beta_2sls_nc, _ = func.two_sls_scratch(Y, D, Z)
print(f"\n  Sanity check: 2SLS no covariates = {beta_2sls_nc[1]:.2f}  vs  Wald = {late_wald:.2f}")

# Robustness: HC1 robust SEs (homoskedastic SEs reported in the table)
se_itt_hc1  = func.ols_se_hc1_scratch(X_itt, resid_itt)
se_ols_hc1  = func.ols_se_hc1_scratch(X_ols, resid_ols)

# HC1 for 2SLS
PZ_  = np.column_stack([np.ones(n), Z, W])
PZ_  = PZ_ @ np.linalg.pinv(PZ_.T @ PZ_) @ PZ_.T
Xf_  = np.column_stack([np.ones(n), D, W])
Xh_  = PZ_ @ Xf_
res_ = Y - Xf_ @ beta_2sls
bread_ = np.linalg.pinv(Xh_.T @ Xf_)
meat_  = (Xh_ * (res_ ** 2)[:, None]).T @ Xh_
V_2sls_hc1 = bread_ @ meat_ @ bread_ * (n / (n - Xf_.shape[1]))
se_2sls_hc1 = np.sqrt(np.diag(V_2sls_hc1))

print("\n  Robustness (HC1 robust SEs):")
print(f"    OLS Y~D     : SE_homo={se_ols[1]:.2f}, SE_HC1={se_ols_hc1[1]:.2f}")
print(f"    ITT         : SE_homo={se_itt[1]:.2f}, SE_HC1={se_itt_hc1[1]:.2f}")
print(f"    2SLS (on D) : SE_homo={se_D:.2f}, SE_HC1={se_2sls_hc1[1]:.2f}")


############## Summary plot for Exercise 8
fig, ax = plt.subplots(figsize=(7, 4.5))
labels    = ["ITT\n(OLS Y on Z)", "LATE\n(Wald)", "2SLS\n(with covariates)"]
estimates = [beta_itt[1], late_wald, coef_D]
errors    = [1.96 * se_itt[1], 1.96 * late_se, 1.96 * se_D]
colors    = ["tab:blue", "tab:orange", "tab:green"]

ax.bar(labels, estimates, yerr=errors, capsize=8, color=colors,
       edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_ylabel("Estimated effect on outcome")
ax.set_title("IV estimates with 95% confidence intervals")
for i, v in enumerate(estimates):
    ax.text(i, v + 100, f"{v:,.0f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ex8_iv_estimates.png"), dpi=150)
plt.close()


############## Final summary table
summary = pd.DataFrame({
    "Estimator":   ["ITT", "LATE (Wald)", "2SLS (with cov.)"],
    "Coefficient": [beta_itt[1], late_wald, coef_D],
    "SE":          [se_itt[1],   late_se,   se_D],
    "t-stat":      [t_itt,       late_t,    t_D],
    "p-value":     [p_itt,       late_p,    p_D]
})
print("\n" + "="*60)
print("SUMMARY - Exercise 8")
print("="*60)
print(summary.round(3).to_string(index=False))


############## EXERCISE 9 - Comparing Identification Strategies
print("\n" + "#"*70)
print("# EXERCISE 9 - Part I (Regions B,C) vs Part II (Region A)")
print("#"*70)

# Recompute Part I estimates on Regions B and C
# (after overlap trimming, as in Part I solution)
import statsmodels.api as sm

# Convert dummy columns to int (needed for logit)
df_full = df.copy()
df_full[edu_dummies.columns]    = df_full[edu_dummies.columns].astype(int)
df_full[region_dummies.columns] = df_full[region_dummies.columns].astype(int)

covariates_prop = covariates + ["offer"] + list(region_dummies.columns)
X_prop = sm.add_constant(df_full[covariates_prop])
ps_model = sm.Logit(df_full["takeup"], X_prop).fit(disp=0)
df_full["propensity_score"] = ps_model.predict(X_prop)

# Drop Region A and trim to common support of B and C
df_BC = df_full[df_full["region"] != "A"].copy()
min_p = max(df_BC[df_BC["region"] == "C"]["propensity_score"].min(),
            df_BC[df_BC["region"] == "B"]["propensity_score"].min())
max_p = min(df_BC[df_BC["region"] == "C"]["propensity_score"].max(),
            df_BC[df_BC["region"] == "B"]["propensity_score"].max())
df_BC = df_BC[(df_BC["propensity_score"] > min_p) &
              (df_BC["propensity_score"] < max_p)]
print(f"\nN (B+C, trimmed): {len(df_BC)}")

covariates_ols = covariates + list(region_dummies.columns)

# Mean difference
md = func.mean_diff(df_BC["takeup"], df_BC["outcome"])

# OLS with covariates
ols_res = func.ols_regression(df_BC[covariates_ols + ["takeup", "outcome"]], "outcome")
ols_coef = ols_res.loc[ols_res["Feature"] == "takeup", "Coefficient"].values[0]
ols_se   = ols_res.loc[ols_res["Feature"] == "takeup", "Standard Error"].values[0]

# IPW
ipw = func.my_ipw(exog=df_BC[covariates_ols], outcome=df_BC["outcome"],
                  treat=df_BC["takeup"], display=False, boot=100)
ipw_coef = ipw["coef"].values[0]
ipw_se   = ipw["se"].values[0]

# Doubly robust
dr  = func.my_dr_custom(df_BC[covariates_ols + ["takeup", "outcome"]],
                        "outcome", "takeup", display=False, boot=100)
dr_coef = dr["coef"].values[0]
dr_se   = dr["se"].values[0]

############## Combined comparison table
comparison = pd.DataFrame({
    "Part":      ["I", "I", "I", "I", "II", "II", "II"],
    "Estimator": ["Mean diff (raw)", "OLS (with cov.)", "IPW", "Doubly Robust",
                  "ITT", "LATE (Wald)", "2SLS (with cov.)"],
    "Sample":    ["B+C", "B+C", "B+C", "B+C", "A", "A", "A"],
    "Estimand":  ["ATE*", "ATE", "ATE", "ATE", "ITT", "LATE", "LATE"],
    "Coeff":     [md["mean_dif"], ols_coef, ipw_coef, dr_coef,
                  beta_itt[1],    late_wald, coef_D],
    "SE":        [md["se"], ols_se, ipw_se, dr_se,
                  se_itt[1], late_se, se_D]
})
print("\n" + "="*70)
print("PART I vs PART II - combined estimates")
print("="*70)
print(comparison.round(2).to_string(index=False))
print("* Mean difference assumes random assignment, which does not hold in B/C.")


############## Plot: Part I vs Part II side-by-side
fig, ax = plt.subplots(figsize=(9, 5))
labels    = comparison["Estimator"].tolist()
estimates = comparison["Coeff"].values
errors    = 1.96 * comparison["SE"].values
colors    = ["tab:gray", "tab:cyan", "tab:cyan", "tab:cyan",
             "tab:blue", "tab:orange", "tab:green"]

bars = ax.bar(labels, estimates, yerr=errors, capsize=6, color=colors,
              edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(3.5, color="red", linewidth=1, linestyle="--", alpha=0.6)
ax.text(1.5, ax.get_ylim()[1]*0.95, "Part I (Regions B+C)",
        ha="center", fontweight="bold", fontsize=10)
ax.text(5,   ax.get_ylim()[1]*0.95, "Part II (Region A)",
        ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("Estimated effect on outcome")
ax.set_title("Part I vs Part II: causal estimates with 95% CIs")
plt.xticks(rotation=20, ha="right")
for i, v in enumerate(estimates):
    ax.text(i, v + 80, f"{v:,.0f}", ha="center", fontsize=8.5, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ex9_comparison.png"), dpi=150)
plt.close()

print(f"\nAll plots saved to: {PLOTS_DIR}/")