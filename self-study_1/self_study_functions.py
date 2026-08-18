"Author: Iacopo Ruggero"


import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# =========================================================
# 1. INSPECTION HELPERS
# =========================================================

def dataset_overview(df):
    # Quick structural overview: shape, column names, dtypes, duplicates, first rows
    print("\n---------------------- DATASET OVERVIEW -----------------------\n")
    print("Shape (observations, variables):")
    print(df.shape)
    print("\nVariables:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\nNumber of duplicated rows:")
    print(df.duplicated().sum())
    print("\nFirst 5 observations:")
    print(df.head())
    print("\n--------------------------------------------------------------\n")


def variables_summary(df):
    # Returns a summary DataFrame with missingness and uniqueness info for every column
    summary_dict = {}
    for col in df.columns:
        series = df[col]
        n_obs = len(series)
        n_missing = series.isna().sum()
        summary_dict[col] = {
            "dtype":       series.dtype,
            "n_obs":       n_obs,
            "n_missing":   n_missing,
            "missing_pct": (n_missing / n_obs) * 100,  # share of missing values
            "n_unique":    series.nunique(dropna=True)  # excludes NaN from count
        }
    return pd.DataFrame(summary_dict).T


def inspect_categorical_dummies(df, varname):
    # Frequency table for a categorical or binary variable, sorted descending
    # dropna=False ensures missing values are also counted and visible
    series = df[varname]
    table = series.value_counts(dropna=False).reset_index()
    table.columns = [varname, "count"]
    return table.sort_values("count", ascending=False)


def inspect_numeric(df, varname):
    # Summary stats for numeric variables focusing on potential data issues:
    # negatives, zeros, and decimals (unexpected for integer-coded variables)
    series = pd.to_numeric(df[varname], errors="coerce")  # coerce non-numeric to NaN
    return pd.DataFrame({
        "statistic": ["n_obs", "min", "max", "n_negative", "n_zero", "n_decimal"],
        "value": [
            len(series),
            series.min(),
            series.max(),
            (series < 0).sum(),                          # flag unexpected negatives
            (series == 0).sum(),                         # flag potential miscodes as zero
            ((series % 1 != 0) & series.notna()).sum()   # flag unexpected decimals
        ]
    })


def inspect_percentage(df, varname):
    # For variables that should lie in [0, 100]: flags out-of-range values
    series = pd.to_numeric(df[varname], errors="coerce")
    return pd.DataFrame({
        "statistic": ["n_obs", "min", "max", "n_below_0", "n_above_100"],
        "value": [
            len(series),
            series.min(),
            series.max(),
            (series < 0).sum(),    # below valid range
            (series > 100).sum()   # above valid range
        ]
    })


def stars(p):
    # Significance stars: *** p<0.01, ** p<0.05, * p<0.10
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


# =========================================================
# 2. PLOT HELPERS (Question 2)
# =========================================================

def plot_binary_by_region(df, varname, output_folder):
    # Bar chart of regional means for a binary variable (e.g. take-up rate by region)
    rates = df.groupby("region")[varname].mean().reindex(["A", "B", "C"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates.index, rates.values)
    ax.set_title(f"{varname} by region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Mean / share")
    fig.savefig(output_folder / f"{varname}_by_region.png", dpi=180, bbox_inches="tight")
    plt.close()  # close figure to free memory


def plot_numeric_hist_by_region(df, varname, output_folder, bins=30):
    # Overlaid density histograms by region: useful for spotting distributional differences
    fig, ax = plt.subplots(figsize=(8, 5))
    for region in ["A", "B", "C"]:
        values = df.loc[df["region"] == region, varname].dropna()
        ax.hist(values, bins=bins, density=True, alpha=0.5, label=region)  # density=True for comparability
    ax.set_title(f"{varname} distribution by region")
    ax.set_xlabel(varname)
    ax.set_ylabel("Density")
    ax.legend(title="Region")
    fig.savefig(output_folder / f"histogram_of_{varname}.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_numeric_box_by_region(df, varname, output_folder):
    # Boxplot by region: highlights median, spread, and outliers side by side
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=varname, by="region", ax=ax)
    ax.set_title(f"{varname} by region")
    ax.set_xlabel("Region")
    ax.set_ylabel(varname)
    plt.suptitle("")  # suppress the automatic pandas figure-level title
    fig.savefig(output_folder / f"boxplot_of_{varname}.png", dpi=180, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. COVARIATE BALANCE (Question 4)
# =========================================================

def covariate_balance_table(df, d_var, x_vars, region=None):
    # Computes SMD = (mean_treated - mean_control) / pooled_std for each covariate.
    # SMD is scale-free, allowing comparison across variables with different units.
    # |SMD| < 0.1 is the conventional threshold for acceptable balance.
    #
    # For the "education" categorical variable, each level is expanded into a binary
    # indicator (share of treated/control with that level) and SMD is computed per
    # dummy — this is the standard approach for categorical variables in balance tables.
    # "region" is still skipped because it is a stratification variable, not a confounder
    # in the within-region analysis.

    sub = df.copy()
    if region is not None:
        sub = sub[sub["region"] == region]  # optionally restrict to one region

    treated = sub[sub[d_var] == 1]
    control = sub[sub[d_var] == 0]

    rows = []

    for var in x_vars:
        if var == "region":
            continue  # region is a stratification variable — skip

        if var == "education":
            # Expand to dummies and compute SMD for each level
            edu_levels = ["low", "medium", "high", "tertiary"]
            for level in edu_levels:
                col_t = (treated["education"] == level).astype(float)
                col_c = (control["education"] == level).astype(float)
                mean_t = col_t.mean()
                mean_c = col_c.mean()
                pooled_std = np.sqrt((col_t.var(ddof=1) + col_c.var(ddof=1)) / 2)
                smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else np.nan
                rows.append({
                    "variable":     f"education={level}",
                    "mean_treated": round(mean_t, 3),
                    "mean_control": round(mean_c, 3),
                    "smd":          round(smd, 3),
                    "abs_smd":      round(abs(smd), 3) if not np.isnan(smd) else np.nan
                })
            continue

        col_t = pd.to_numeric(treated[var], errors="coerce").dropna()
        col_c = pd.to_numeric(control[var], errors="coerce").dropna()

        mean_t = col_t.mean()
        mean_c = col_c.mean()
        # pooled SD: average of treated and control variances (not weighted by group size)
        pooled_std = np.sqrt((col_t.var(ddof=1) + col_c.var(ddof=1)) / 2)
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else np.nan

        rows.append({
            "variable":     var,
            "mean_treated": round(mean_t, 3),
            "mean_control": round(mean_c, 3),
            "smd":          round(smd, 3),
            "abs_smd":      round(abs(smd), 3)
        })

    # Sort by absolute SMD descending so the most imbalanced variables appear first
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False)


# =========================================================
# 4. OVERLAP STATISTICS (Question 2)
# =========================================================

def ps_overlap_stats(ps, d, region_label=None):
    # Computes common support bounds and the share of treated units outside control PS range
    # Poor overlap means some treated units have no comparable controls -> PS weights unstable

    ps = np.array(ps)
    d  = np.array(d)

    ps_t = ps[d == 1]  # PS values for treated units
    ps_c = ps[d == 0]  # PS values for control units

    # Common support: intersection of the two PS ranges
    support_min = max(ps_t.min(), ps_c.min())
    support_max = min(ps_t.max(), ps_c.max())

    results = {
        "control_ps_min":                 round(float(ps_c.min()), 4),
        "control_ps_max":                 round(float(ps_c.max()), 4),
        "treated_ps_min":                 round(float(ps_t.min()), 4),
        "treated_ps_max":                 round(float(ps_t.max()), 4),
        "common_support_min":             round(float(support_min), 4),
        "common_support_max":             round(float(support_max), 4),
        # share of all observations within common support
        "share_on_common_support":        round(float(((ps >= support_min) & (ps <= support_max)).mean()), 4),
        # share of treated units that fall outside the control PS range entirely
        "share_treated_off_ctrl_support": round(float(((ps_t < ps_c.min()) | (ps_t > ps_c.max())).mean()), 4),
    }

    df_out = pd.DataFrame.from_dict(results, orient="index", columns=["value"])
    df_out.index.name = "statistic"

    if region_label is not None:
        print(f"\nOverlap statistics - Region {region_label}")
        print("-" * 40)
    print(df_out.to_string())

    return df_out


def plot_covariate_balance(df, d_var, x_vars, output_folder, regions=["A", "B", "C"]):
    """
    Dot plot of absolute SMD by covariate for each region.
    Each region gets a differently shaped/colored marker.
    Vertical dashed line at 0.1 marks the conventional balance threshold.
    Education is included: covariate_balance_table expands it into per-level dummies.
    """
    # Pass the full x_vars list (including "education"); the table function handles expansion.
    # Only "region" is excluded here — it is a stratification variable, not a confounder.
    balance_vars = [v for v in x_vars if v != "region"]

    # collect SMD for each region into a single DataFrame
    records = []
    for r in regions:
        tbl = covariate_balance_table(df, d_var, balance_vars, region=r)
        for _, row in tbl.iterrows():
            records.append({
                "region":   r,
                "variable": row["variable"],
                "abs_smd":  row["abs_smd"]
            })

    plot_df = pd.DataFrame(records)

    # order variables by mean abs_smd across regions (largest imbalance at top)
    var_order = (plot_df.groupby("variable")["abs_smd"]
                        .mean()
                        .sort_values(ascending=True)
                        .index.tolist())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors  = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
    markers = {"A": "o",       "B": "s",        "C": "^"}
    offsets = {"A": -0.15,     "B": 0.0,        "C": 0.15}  # vertical jitter to avoid overlap

    for r in regions:
        sub         = plot_df[plot_df["region"] == r].set_index("variable")
        y_positions = [var_order.index(v) + offsets[r] for v in var_order if v in sub.index]
        x_values    = [sub.loc[v, "abs_smd"] for v in var_order if v in sub.index]
        ax.scatter(x_values, y_positions,
                   label=f"Region {r}", color=colors[r],
                   marker=markers[r], s=60, zorder=3)

    # threshold line at 0.1
    ax.axvline(0.1, color="red", linestyle="--", linewidth=1, label="Threshold (0.1)")

    ax.set_yticks(range(len(var_order)))
    ax.set_yticklabels(var_order)
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title("Covariate balance: treated vs control by region")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / "covariate_balance_dotplot.png", dpi=180, bbox_inches="tight")
    plt.close()


# =========================================================
# 5. OLS / ROBUST SE HELPERS
# =========================================================

def p_value_from_estimate(est, se):
    # Two-sided p-value from a z-statistic (large-sample normal approximation)
    z = est / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def ols_fit(X, y):
    # OLS estimator via the pseudo-inverse: beta = (X'X)^{-1} X'y
    # Pseudo-inverse (pinv) is used instead of direct inverse for numerical stability
    # when X'X is near-singular (e.g. due to multicollinearity or small subsamples)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ (X.T @ y)
    fitted  = X @ beta
    resid   = y - fitted  # OLS residuals
    return beta, resid, XtX_inv, fitted


def robust_se_ols(X, resid, XtX_inv, param_index):
    # HC1 heteroskedasticity-robust (sandwich) standard error for one coefficient
    # HC1 applies a degrees-of-freedom correction: multiplies by n/(n-k)
    # Preferable to default OLS SE when error variances are non-constant
    n, k = X.shape
    # Vectorised meat computation: equivalent to sum_i e_i^2 * x_i x_i'
    # but avoids the slow Python-level loop over n observations
    scaled = X * resid[:, np.newaxis]          # (n, k): each row i is e_i * x_i
    meat   = scaled.T @ scaled                 # (k, k): sum_i e_i^2 x_i x_i'
    vcov   = XtX_inv @ meat @ XtX_inv          # sandwich formula
    vcov  *= n / (n - k)                       # HC1 degrees-of-freedom correction
    return np.sqrt(vcov[param_index, param_index])  # SE = sqrt of diagonal element


# =========================================================
# 6. MATRIX BUILDERS
# =========================================================

def build_outcome_matrix(data, x_vars, treatment_var):
    # Design matrix for regression adjustment:
    # columns = [intercept, treatment, covariates (education/region dummified)]
    # drop_first=True avoids the dummy variable trap (perfect collinearity with intercept)
    X = pd.get_dummies(data[x_vars].copy(), columns=["education", "region"], drop_first=True)
    X.insert(0, treatment_var, data[treatment_var].astype(float).values)  # treatment in second position
    X.insert(0, "intercept", 1.0)                                         # constant in first position
    return X.to_numpy(dtype=float), X.columns.tolist()  # return array and names for coefficient indexing


def build_covariate_matrix(data, x_vars):
    # Covariate-only design matrix for the outcome models m1(x) and m0(x) in DR
    # No treatment column: these models are fitted separately on treated and control subsamples
    X = pd.get_dummies(data[x_vars].copy(), columns=["education", "region"], drop_first=True)
    X.insert(0, "intercept", 1.0)
    return X  # returned as DataFrame to allow column alignment across subsamples


def build_ps_features(data, x_vars):
    # Preprocessed feature matrix for the propensity score logistic regression.
    # Only variables actually present in x_vars are included in each transformer,
    # so calling this function within a single region (where x_vars excludes "region")
    # correctly omits the region encoder instead of adding a zero-variance dummy column.
    #
    # Numeric vars : imputed with median (robust to outliers) and standardised for logit stability
    # Binary vars  : mode imputation only (no scaling needed for 0/1 indicators)
    # Categorical  : mode imputation + one-hot encoding with drop_first=True

    _numeric_candidates     = ["age", "special_trainings", "baseline_income",
                                "unemp_duration", "language_score"]
    _binary_candidates      = ["gender", "employment_gaps", "immigrant", "disability"]
    _categorical_candidates = ["education", "region"]

    numeric_vars     = [v for v in _numeric_candidates     if v in x_vars]
    binary_vars      = [v for v in _binary_candidates      if v in x_vars]
    categorical_vars = [v for v in _categorical_candidates if v in x_vars]

    transformers = []
    if numeric_vars:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ]),
            numeric_vars
        ))
    if binary_vars:
        transformers.append((
            "bin",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ]),
            binary_vars
        ))
    if categorical_vars:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ]),
            categorical_vars
        ))

    preprocessor = ColumnTransformer(transformers)
    return preprocessor.fit_transform(data[x_vars].copy())


def estimate_propensity_scores(data, x_vars, d_var):
    # Fits a logistic regression and returns P(D=1|X) for each observation.
    # Scores are trimmed away from 0 and 1 to avoid division by zero in IPW and DR.
    X_ps = build_ps_features(data, x_vars)
    D_ps = data[d_var].to_numpy().astype(int)

    model = LogisticRegression(max_iter=5000)   # high max_iter for convergence on scaled data
    model.fit(X_ps, D_ps)

    ps = model.predict_proba(X_ps)[:, 1]        # column 1 = P(D=1|X)
    return np.clip(ps, 1e-6, 1 - 1e-6)          # trim for numerical stability in weight computation


# =========================================================
# 7. ESTIMATORS (Question 3)
# =========================================================

def raw_mean_difference(y, d):
    # Difference in sample means: E[Y|D=1] - E[Y|D=0]
    # Valid ATE estimator only under unconditional exogeneity (full randomization)
    # SE uses Welch formula: does not assume equal variances across groups
    y1  = y[d == 1]
    y0  = y[d == 0]
    est = y1.mean() - y0.mean()
    se  = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z, p = p_value_from_estimate(est, se)
    return {"estimate": est, "se": se, "z": z, "p_value": p}


def _regression_adjustment_point_estimate(data, y_var, d_var, x_vars):
    """
    Point estimate for the imputation-style regression adjustment estimator:
        tau_RA = (1/n) * sum_i [ m1(X_i) - m0(X_i) ]
    where m1 and m0 are fitted separately on treated and control subsamples.
    """
    treated_data = data[data[d_var] == 1].copy()
    control_data = data[data[d_var] == 0].copy()

    if treated_data.empty or control_data.empty:
        raise ValueError("Both treated and control observations are required.")

    X1_df = build_covariate_matrix(treated_data, x_vars)
    X0_df = build_covariate_matrix(control_data, x_vars)

    y1 = treated_data[y_var].to_numpy(dtype=float)
    y0 = control_data[y_var].to_numpy(dtype=float)

    # Align columns across subsamples
    X1_df, X0_df = X1_df.align(X0_df, join="outer", axis=1, fill_value=0)

    beta1, _, _, _ = ols_fit(X1_df.to_numpy(dtype=float), y1)
    beta0, _, _, _ = ols_fit(X0_df.to_numpy(dtype=float), y0)

    # Predict potential outcomes for the full sample
    X_full = build_covariate_matrix(data, x_vars)
    X_full_1 = X_full.reindex(columns=X1_df.columns, fill_value=0).to_numpy(dtype=float)
    X_full_0 = X_full.reindex(columns=X0_df.columns, fill_value=0).to_numpy(dtype=float)

    m1 = X_full_1 @ beta1
    m0 = X_full_0 @ beta0

    return float(np.mean(m1 - m0))


def regression_adjustment(data, y_var, d_var, x_vars, n_bootstrap=500, seed=42):
    """
    Regression adjustment ATE with bootstrap standard error.

    Why bootstrap here:
    The old SE based only on var(m1 - m0)/n is too small because it ignores
    uncertainty from estimating the outcome models. Bootstrapping the full
    estimator is a much safer way to get inference for this imputation estimator.
    """
    est = _regression_adjustment_point_estimate(data, y_var, d_var, x_vars)

    rng = np.random.default_rng(seed)
    d = data[d_var].to_numpy()
    treated_idx = np.where(d == 1)[0]
    control_idx = np.where(d == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        raise ValueError("Both treated and control observations are required.")

    boot_estimates = []

    # Stratified bootstrap: preserve treated/control sample sizes in each resample
    for _ in range(n_bootstrap):
        idx_t = rng.choice(treated_idx, size=len(treated_idx), replace=True)
        idx_c = rng.choice(control_idx, size=len(control_idx), replace=True)
        idx = np.concatenate([idx_t, idx_c])

        boot_df = data.iloc[idx].copy()

        boot_est = _regression_adjustment_point_estimate(
            boot_df, y_var, d_var, x_vars
        )
        boot_estimates.append(boot_est)

    se = float(np.std(boot_estimates, ddof=1))
    z, p = p_value_from_estimate(est, se)

    return {
        "estimate": est,
        "se": se,
        "z": z,
        "p_value": p,
        "boot_estimates": boot_estimates
    }

def ipw_ate(data, y_var, d_var, x_vars):
    # Horvitz-Thompson IPW estimator: reweights observations by inverse PS
    # Treated units get weight 1/p(x), controls get weight 1/(1-p(x))
    # Creates a pseudo-population where treatment is approximately independent of X
    # SE computed from the influence function (sample variance of the psi vector)
    y  = data[y_var].to_numpy(dtype=float)
    d  = data[d_var].to_numpy(dtype=float)
    ps = estimate_propensity_scores(data, x_vars, d_var)

    psi = d * y / ps - (1 - d) * y / (1 - ps)  # IPW influence function
    est = psi.mean()
    se  = np.sqrt(np.var(psi, ddof=1) / len(psi))
    z, p = p_value_from_estimate(est, se)
    return {"estimate": est, "se": se, "z": z, "p_value": p}


def ipw_ate_normalized_bootstrap(data, y_var, d_var, x_vars, n_bootstrap=500, seed=42):
    """
    Normalized (Hajek) IPW ATE estimator with bootstrap SE.
    Resamples the full dataset n_bootstrap times, re-estimates PS and
    weighted means each time, and uses the SD of bootstrap estimates as SE.
    """
    rng = np.random.default_rng(seed)
    y   = data[y_var].to_numpy(dtype=float)
    d   = data[d_var].to_numpy(dtype=float)
    n   = len(data)

    def hajek_estimate(y_, d_, ps_):
        # normalized weights within each group
        w1 = (d_ / ps_)
        w0 = ((1 - d_) / (1 - ps_))
        w1 = w1 / w1[d_ == 1].sum()
        w0 = w0 / w0[d_ == 0].sum()
        return (w1 * y_).sum() - (w0 * y_).sum()

    # point estimate on the original sample
    ps_orig = estimate_propensity_scores(data, x_vars, d_var)
    est     = hajek_estimate(y, d, ps_orig)

    # bootstrap loop
    boot_estimates = []
    for _ in range(n_bootstrap):
        # resample rows with replacement
        idx      = rng.integers(0, n, size=n)
        boot_df  = data.iloc[idx].copy()

        # re-estimate PS on the bootstrap sample (important: not just resample weights)
        ps_boot  = estimate_propensity_scores(boot_df, x_vars, d_var)
        y_boot   = boot_df[y_var].to_numpy(dtype=float)
        d_boot   = boot_df[d_var].to_numpy(dtype=float)

        boot_estimates.append(hajek_estimate(y_boot, d_boot, ps_boot))

    # SE = standard deviation of bootstrap distribution
    se   = np.std(boot_estimates, ddof=1)
    z, p = p_value_from_estimate(est, se)

    return {
        "estimate":         est,
        "se":               se,
        "z":                z,
        "p_value":          p,
        "boot_estimates":   boot_estimates  # keep for diagnostics if needed
    }


def doubly_robust_ate(data, y_var, d_var, x_vars):
    # Augmented IPW (AIPW): combines outcome models m1(x), m0(x) with PS weighting
    # Consistent if EITHER the outcome model OR the PS model is correctly specified
    # Outcome models fitted separately on treated and control to avoid pooling assumptions
    # Column alignment handles cases where a dummy level is absent in one subsample
    y  = data[y_var].to_numpy(dtype=float)
    d  = data[d_var].to_numpy(dtype=float)
    ps = estimate_propensity_scores(data, x_vars, d_var)

    treated_data = data[data[d_var] == 1].copy()
    control_data = data[data[d_var] == 0].copy()

    X1_df = build_covariate_matrix(treated_data, x_vars)  # features for treated outcome model
    X0_df = build_covariate_matrix(control_data,  x_vars)  # features for control outcome model
    y1    = treated_data[y_var].to_numpy(dtype=float)
    y0    = control_data[y_var].to_numpy(dtype=float)

    # Align columns: one subsample may lack a dummy level present in the other
    X1_df, X0_df = X1_df.align(X0_df, join="outer", axis=1, fill_value=0)

    beta1, _, _, _ = ols_fit(X1_df.to_numpy(dtype=float), y1)  # outcome model for treated
    beta0, _, _, _ = ols_fit(X0_df.to_numpy(dtype=float), y0)  # outcome model for controls

    # Predict m1(x) and m0(x) for the full sample using aligned column structure
    X_full   = build_covariate_matrix(data, x_vars)
    X_full_1 = X_full.reindex(columns=X1_df.columns, fill_value=0).to_numpy(dtype=float)
    X_full_0 = X_full.reindex(columns=X0_df.columns, fill_value=0).to_numpy(dtype=float)

    m1 = X_full_1 @ beta1  # predicted E[Y(1)|X]
    m0 = X_full_0 @ beta0  # predicted E[Y(0)|X]

    # AIPW influence function: outcome model difference + IPW residual correction term
    psi = m1 - m0 + d * (y - m1) / ps - (1 - d) * (y - m0) / (1 - ps)
    est = psi.mean()
    se  = np.sqrt(np.var(psi, ddof=1) / len(psi)) 
    z, p = p_value_from_estimate(est, se)
    return {"estimate": est, "se": se, "z": z, "p_value": p}