# Self Study Part 1

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats


def plot_histograms(df):
    """
    Plots histograms for all numerical columns in the given DataFrame to visually inspect for outliers.

    Parameters:
        df : pd.DataFrame
            The input DataFrame containing numerical columns.

    Returns:
        None
            Displays the histograms for numerical columns.
    """
    
    # Select only numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        print("No numerical columns found in the dataset.")
        return

    # Set seaborn style
    sns.set_style("whitegrid")

    # Determine the number of rows and columns for subplots
    num_plots = len(num_cols)
    cols = 3  # Number of columns in the plot grid
    rows = (num_plots // cols) + (num_plots % cols > 0)  # Ensure enough rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # Flatten in case of single row/column
    
    # Loop through numerical columns and plot histograms
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color="royalblue")
        axes[i].set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")
    
    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



def calculate_stats(df):
    """
    Calculate the mean and standard deviation for all numeric columns in a DataFrame
    without using built-in std() function.
    
    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: DataFrame with mean and standard deviation for each column
    """
    stats = {}
    
    for col in df.select_dtypes(include=[np.number]):  # Process only numeric columns
        values = df[col].dropna().values  # Remove NaN values
        n = len(values)
        
        if n == 0:
            continue  # Skip empty columns
        
        # Calculate mean
        mean_value = sum(values) / n
        
        # Calculate standard deviation manually
        variance = sum((x - mean_value) ** 2 for x in values) / n  # Population variance
        std_dev = variance ** 0.5  # Square root of variance
        
        stats[col] = {"Mean": mean_value, "Standard Deviation": std_dev}
    
    return pd.DataFrame(stats).T  # Convert dictionary to DataFrame for better readability


def mean_diff(treatment, outcome):
    '''
    Compute mean difference

    Parameters
        treatment:
            TYPE: pandas.Series
            DESCRIPTION: Vector of treatments
        outcome:
            TYPE: pandas.Series
            DESCRIPTION: Vector of outcomes

    Returns
    Returns mean difference, standard error, t statistic, and p-value
    '''
    # Produce vector of outcomes for treated and controls
    treated = outcome.loc[treatment == 1]
    controls = outcome.loc[treatment == 0]

    # Compute mean difference
    mean_dif = sum(treated) / len(treated) - sum(controls) / len(controls)

    # Compute variance manually for treated and controls
    treated_variance = np.sum((treated - np.mean(treated)) ** 2) / len(treated)
    controls_variance = np.sum((controls - np.mean(controls)) ** 2) / len(controls)

    # Compute standard error (independent samples)
    se = np.sqrt(treated_variance / len(treated) + controls_variance / len(controls))

    # Compute t statistic
    t = mean_dif / se

    # Compute p-value
    # Compute degrees of freedom
    df = (treated_variance / len(treated) + controls_variance / len(controls)) ** 2 / \
         ((treated_variance / len(treated)) ** 2 / (len(treated) - 1) + (controls_variance / len(controls)) ** 2 / (len(controls) - 1))
    p = (1 - scipy.stats.t.cdf(abs(t), df)) * 2

    # Return results
    return {'mean_dif': mean_dif, 'se': se, 't': t, 'p': p, 'df': df}


def ols_regression(df, y_col):
    """
    Perform OLS regression using matrix operations (NumPy) on a Pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing independent variables and the dependent variable.
        y_col (str): Name of the column representing the dependent variable.

    Returns:
        dict: Coefficients and standard errors.
    """
    # Extract y (dependent variable) and X (independent variables)
    y = df[y_col].values.reshape(-1, 1)  # Convert to column vector
    X = df.drop(columns=[y_col]).values  # Drop dependent variable to get predictors
    
    # Add intercept (column of 1s)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  

    n, p = X.shape  # n = observations, p = predictors (including intercept)

    # Compute OLS estimates: beta = (X^T X)^(-1) X^T y
    XTX = X.T @ X  # X^T X
    XTX_inv = np.linalg.pinv(XTX)  # (X^T X)^(-1)
    XTy = X.T @ y  # X^T y
    beta = XTX_inv @ XTy  # Beta coefficients

    # Compute residuals and variance
    y_hat = X @ beta  # Predicted values
    residuals = y - y_hat  # Residuals
    residual_variance = (residuals.T @ residuals) / (n - p)  # σ² = RSS / (n - p)

    # Compute standard errors: SE = sqrt(diag(sigma^2 * (X^T X)^(-1)))
    variance_cov_matrix = residual_variance * XTX_inv  # σ² * (X^T X)^(-1)
    standard_errors = np.sqrt(np.diag(variance_cov_matrix))  # Extract diagonal elements

    # Create result dictionary with feature names
    feature_names = ["Intercept"] + list(df.drop(columns=[y_col]).columns)
    
    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": beta.flatten(),
        "Standard Error": standard_errors.flatten()
    })


def logistic_regression(df, y_col, selected_columns):
    """
    Perform logistic regression using statsmodels and predict the propensity score.

    Args:
        df (pd.DataFrame): DataFrame containing the features and target variable.
        y_col (str): The name of the target column.
        selected_columns (list): List of predictor variable names.

    Returns:
        dict: Coefficients, standard errors, model summary, and propensity scores.
    """
    # Define X (features) and y (target variable)
    X = df[selected_columns]
    y = df[y_col]

    # Add a constant (intercept) column to X
    X = sm.add_constant(X)

    # Fit the logistic regression model using statsmodels (for standard errors)
    model = sm.Logit(y, X)
    result = model.fit()

    # Get coefficients and standard errors
    coefficients = result.params
    standard_errors = result.bse

    # Predict the propensity scores (probabilities)
    propensity_scores = result.predict(X)

    # Display model summary
    model_summary = result.summary()

    # Convert coefficients and standard errors to a dictionary
    output = {
        'coefficients': coefficients,
        'standard_errors': standard_errors,
        'summary': model_summary,
        'propensity_scores': propensity_scores
    }

    return output



# own ipw function
def my_ipw(exog, outcome, treat, intercept=True, display=True,
           boot=100):
    """
    IPW estimation.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: matrix of covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treat : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    intercept : TYPE: boolean, optional
        DESCRIPTION: if intercept should be used.
        The default is True.
    boot : TYPE: integer, optional
        DESCRIPTION: Number of bootstrap replications for inference.
        The default is 100.
    lasso : TYPE: boolean, optional
        DESCRIPTION: if lasso should be used for propensity score.
        The default is False.
    cval : TYPE: boolean, optional
        DESCRIPTION: if cross-validation should be used for lasso.
        The default is False.
    folds : TYPE: integer, optional
        DESCRIPTION: number of folds for cross-validation.
        The default is 10.

    Returns
    -------
    result: ipw estimates with standard errors
    """
    # estimate ate via ipw and pass on the keyword arguments for estimation
    ate, pscores = ate_ipw(
        exog=exog, outcome=outcome, treat=treat, intercept=intercept)
    # estimate SE for ATE using bootstrapping and pass on keyword arguments
    ate_se = ate_ipw_boot(exog=exog, outcome=outcome, treat=treat,
                          intercept=intercept, reps=boot)
    # compute t and p values
    tval = ate / ate_se
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    pval = stats.t.sf(np.abs(tval), (exog.shape[0] - exog.shape[1])) * 2
    # combine the results as pd.DataFrame and name the corresponding values
    result = pd.DataFrame([ate, ate_se, tval, pval],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=[treat.name]).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, print the results (\n inserts a line break)
        print('IPW Estimation Results:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the result
    return result


# ipw formula
def ipw_formula(treat, outcome, prob, normalize=True):
    """
    IPW formula.

    Parameters
    ----------
    treat : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    prob : TYPE: pd.Series / array
        DESCRIPTION: vector of probabilities

    Returns
    -------
    result: ATE estimate according to IPW
    """
    # compute ate based on ipw formula, take the mean using numpy
    # This is the standard version - known to have bad  small sample properties
    if not normalize:
        ate = np.mean(treat * outcome / prob
                      - (1 - treat) * outcome / (1 - prob))
    else:
        # We can achieve better small sample properties if we normalize
        # the weights for treated and controls
        w_1 = treat / prob
        w_0 = (1 - treat) / (1 - prob)
        w_1 = w_1 / np.sum(w_1)
        w_0 = w_0 / np.sum(w_0)
        ate = np.sum(w_1 * outcome) - np.sum(w_0 * outcome)
    # return the result
    return ate


# ipw ate estimation
def ate_ipw(exog, outcome, treat,
            intercept=True):
    """
    ATE estimation by IPW.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: matrix of covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treat : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    intercept : TYPE: boolean, optional
        DESCRIPTION: if intercept should be used.
        The default is True.
    lasso : TYPE: boolean, optional
        DESCRIPTION: if lasso should be used for propensity score.
        The default is False.
    cval : TYPE: boolean, optional
        DESCRIPTION: if cross-validation should be used for lasso.
        The default is False.
    folds : TYPE: integer, optional
        DESCRIPTION: number of folds for cross-validation.
        The default is 10.

    Returns
    -------
    result: ATE estimate according to IPW with logit propensity scores
    """
    # estimate propensity scores via logit and extract the predictions
    # use the Logit() function from the statsmodels module
    # chekc if intercept should be included
    # the following condition checks implicitly if intercept == True
    if intercept:
        pscores = sm.Logit(endog=treat, exog=sm.add_constant(exog)).fit(
            disp=0).predict()
    else:
        # if False, no intercept when estimating pscores
        pscores = sm.Logit(endog=treat, exog=exog).fit(disp=0).predict()
    # finally compute ate based on ipw formula and feed in the pscores
    ate = ipw_formula(treat=treat, outcome=outcome, prob=pscores)
    # return the result
    return ate, pscores


# standard errors via bootstrap for ipw ate estimate
def ate_ipw_boot(exog, outcome, treat, intercept=True,
                 reps=100):
    """
    Bootstrap Standard Errors for IPW Estimation.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: matrix of covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treat : TYPE: pd.Series
        DESCRIPTION: vector of treatments
    intercept : TYPE: boolean, optional
        DESCRIPTION: if intercept should be used.
        The default is True.
    reps : TYPE: integer, optional
        DESCRIPTION: Number of bootstrap replications for inference.
        The default is 100.
    lasso : TYPE: boolean, optional
        DESCRIPTION: if lasso should be used for propensity score.
        The default is False.
    cv : TYPE: boolean, optional
        DESCRIPTION: if cross-validation should be used for lasso.
        The default is False.
    folds : TYPE: integer, optional
        DESCRIPTION: number of folds for cross-validation.
        The default is 10.

    Returns
    -------
    result: bootstrap standard error for ATE
    """
    # create storage for the bootstrap estimates as an empty dictionary
    # to easily fill in values including the corresponding names
    ate_boot = []
    # Initialize the numpy random number generator for replicability of results
    rng = np.random.default_rng(12345)
    # loop over replications, use the range function: 0,...,reps-1
    for rep_id in range(reps):
        boot_sample = rng.choice(exog.index, size=exog.shape[0], replace=True)
        exog_boot = exog.loc[boot_sample, :]
        outcome_boot = outcome.loc[boot_sample]
        treat_boot = treat.loc[boot_sample]


        try:
            # Compute ATE using IPW on bootstrap sample
            ate_val, _ = ate_ipw(exog=exog_boot, outcome=outcome_boot,
                                 treat=treat_boot, intercept=intercept)
            ate_boot.append(ate_val)
    
        except np.linalg.LinAlgError:
            # Skip this replication if singular matrix occurs
            print(f"Warning: Singular matrix at bootstrap rep {rep_id}, skipping.")
            continue
        except Exception as e:
            # Catch other unexpected errors
            print(f"Warning: Error at bootstrap rep {rep_id}: {e}, skipping.")
            continue

    if len(ate_boot) == 0:
        raise ValueError("All bootstrap replications failed. Check covariates for multicollinearity or missing values.")

    # compute standard error for ate as standard deviation of all ate estimates
    # take the values from the dictionary and convert them to a list first
    # take the standard deviation using the numpy module
    ate_se = np.std(ate_boot)
    # return the result
    return ate_se


# Outcome regression using OLS (fits separate models for treated/control)
def outcome_regression_custom(df, y_col, treat_col):
    """
    Fit outcome models separately for treated and control using OLS.

    Returns predicted outcomes for all units under both treatment and control.
    """
    # Split treated and control
    df_treat = df[df[treat_col]==1]
    df_control = df[df[treat_col]==0]

    # Fit OLS for treated
    res_treat = ols_regression(df_treat, y_col)
    beta_treat = res_treat['Coefficient'].values.reshape(-1,1)

    # Fit OLS for control
    res_control = ols_regression(df_control, y_col)
    beta_control = res_control['Coefficient'].values.reshape(-1,1)

    # Prepare design matrix for all units
    X_all = np.hstack([np.ones((df.shape[0], 1)), df.drop(columns=[y_col]).values])

    # Predicted outcomes for all units under treatment/control
    mu1_hat = X_all @ beta_treat
    mu0_hat = X_all @ beta_control

    return mu1_hat, mu0_hat



# Doubly robust formula
def dr_formula(treat, outcome, pscores, mu1_hat, mu0_hat):
    """
    Compute ATE using classic Doubly Robust formula.
    """
    dr_scores = treat * (outcome - mu1_hat) / pscores - (1 - treat) * (outcome - mu0_hat) / (1 - pscores) + mu1_hat - mu0_hat
    return np.mean(dr_scores)


# Doubly robust ATE estimation using custom OLS
def ate_dr_custom(df, y_col, treat_col, intercept=True):
    """
    Compute Doubly Robust ATE using custom OLS for outcome regression.
    
    Args:
        df (pd.DataFrame): DataFrame containing covariates, treatment, and outcome
        y_col (str): Name of the outcome column
        treat_col (str): Name of the treatment column
        intercept (bool): Whether to include intercept
    """
    exog_cols = df.drop(columns=[y_col, treat_col]).columns
    exog = df[exog_cols]

    # Estimate propensity scores (still using statsmodels Logit)
    if intercept:
        pscores = sm.Logit(endog=df[treat_col], exog=sm.add_constant(exog)).fit(disp=0).predict()
    else:
        pscores = sm.Logit(endog=df[treat_col], exog=exog).fit(disp=0).predict()

    # Estimate outcome models using custom OLS
    mu1_hat, mu0_hat = outcome_regression_custom(df[[*exog_cols, y_col, treat_col]], y_col, treat_col)

    # Compute DR ATE
    ate = dr_formula(df[treat_col].values, df[y_col].values, pscores, mu1_hat, mu0_hat)
    
    return ate, pscores, mu1_hat, mu0_hat


# Bootstrap standard errors for Doubly Robust ATE using custom OLS
def ate_dr_boot_custom(df, y_col, treat_col, intercept=True, reps=100):
    """
    Bootstrap SEs for Doubly Robust ATE using custom OLS for outcome regression.

    Args:
        df (pd.DataFrame): DataFrame containing covariates, treatment, and outcome
        y_col (str): Name of the outcome column
        treat_col (str): Name of the treatment column
        intercept (bool): Whether to include intercept in OLS
        reps (int): Number of bootstrap replications

    Returns:
        float: bootstrap standard error of the DR ATE
    """
    rng = np.random.default_rng(12345)
    ate_boot = []

    for i in range(reps):
        # Sample indices with replacement
        sample_idx = rng.choice(df.index, size=df.shape[0], replace=True)
        df_boot = df.loc[sample_idx].reset_index(drop=True)

        try:
            # Compute DR ATE using custom OLS
            ate_val, _, _, _ = ate_dr_custom(df_boot, y_col, treat_col, intercept=intercept)
            ate_boot.append(ate_val)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix at bootstrap {i}, skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error at bootstrap {i}: {e}, skipping.")
            continue

    if len(ate_boot) == 0:
        raise ValueError("All bootstrap replications failed.")

    return np.std(ate_boot)


# Full Doubly Robust function using custom OLS
def my_dr_custom(df, y_col, treat_col, intercept=True, display=True, boot=100):
    """
    Doubly Robust ATE estimation with bootstrap SEs using custom OLS for outcome regression.

    Args:
        df (pd.DataFrame): DataFrame containing covariates, treatment, and outcome
        y_col (str): Name of the outcome column
        treat_col (str): Name of the treatment column
        intercept (bool): Include intercept in OLS
        display (bool): Whether to print results
        boot (int): Number of bootstrap replications for SEs

    Returns:
        pd.DataFrame: DR ATE results with coefficient, SE, t-value, p-value
    """
    # Compute ATE and predictions
    ate, pscores, mu1_hat, mu0_hat = ate_dr_custom(df, y_col, treat_col, intercept=intercept)

    # Compute bootstrap standard errors
    ate_se = ate_dr_boot_custom(df, y_col, treat_col, intercept=intercept, reps=boot)

    # t-value and p-value
    tval = ate / ate_se
    n = df.shape[0]
    k = df.shape[1] - 2  # subtract outcome + treatment columns
    pval = stats.t.sf(np.abs(tval), n - k) * 2

    # Format results
    result = pd.DataFrame([ate, ate_se, tval, pval],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=[treat_col]).transpose()

    if display:
        print('Doubly Robust Estimation Results:', '-'*80,
              'Dependent Variable: ' + y_col, '-'*80,
              round(result, 2), '-'*80, '\n\n', sep='\n')
    
    return result


# Part II: from-scratch helpers for IV/instrument validation


def ols_scratch(X, y):
    """
    OLS via normal equations.

    Args:
        X (np.ndarray): design matrix (must include intercept column if desired).
        y (np.ndarray): outcome vector.

    Returns:
        tuple: (beta, residuals, rss).
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    rss = float(resid @ resid)
    return beta, resid, rss


def ols_se_scratch(X, resid):
    """
    Homoskedastic OLS standard errors.

    Args:
        X (np.ndarray): design matrix used in estimation.
        resid (np.ndarray): OLS residuals.

    Returns:
        np.ndarray: standard errors for each coefficient.
    """
    n, k = X.shape
    s2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.pinv(X.T @ X)
    return np.sqrt(s2 * np.diag(XtX_inv))

def ols_se_hc1_scratch(X, resid):
    """
    HC1 (Eicker-Huber-White) robust standard errors for OLS.

    Args:
        X (np.ndarray): design matrix used in estimation.
        resid (np.ndarray): OLS residuals.

    Returns:
        np.ndarray: HC1 standard errors for each coefficient.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    return np.sqrt(np.diag(V))


def welch_t_scratch(a, b):
    """
    Two-sample Welch t-test.

    Args:
        a, b (np.ndarray): samples.

    Returns:
        tuple: (mean_a, mean_b, diff, t-stat, p-value).
    """
    n_a, n_b = len(a), len(b)
    m_a = np.sum(a) / n_a
    m_b = np.sum(b) / n_b
    var_a = np.sum((a - m_a) ** 2) / (n_a - 1)
    var_b = np.sum((b - m_b) ** 2) / (n_b - 1)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    t = (m_a - m_b) / se
    dof = (var_a/n_a + var_b/n_b) ** 2 / (
           (var_a/n_a) ** 2 / (n_a - 1) + (var_b/n_b) ** 2 / (n_b - 1))
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), dof))
    return m_a, m_b, m_a - m_b, t, p


def partial_f_scratch(X_full, X_restricted, y):
    """
    Partial F-test for restricted vs unrestricted OLS (both with intercept).

    Args:
        X_full (np.ndarray): unrestricted design matrix.
        X_restricted (np.ndarray): restricted design matrix.
        y (np.ndarray): outcome vector.

    Returns:
        tuple: (F-stat, p-value).
    """
    n = len(y)
    _, _, rss_u = ols_scratch(X_full, y)
    _, _, rss_r = ols_scratch(X_restricted, y)
    q = X_full.shape[1] - X_restricted.shape[1]
    k = X_full.shape[1]
    F = ((rss_r - rss_u) / q) / (rss_u / (n - k))
    p = 1 - scipy.stats.f.cdf(F, q, n - k)
    return float(F), float(p)


def joint_f_scratch(X_regressors, y):
    """
    Joint F-test that all slopes are zero (intercept added internally).

    Args:
        X_regressors (np.ndarray): regressors without intercept.
        y (np.ndarray): outcome vector.

    Returns:
        tuple: (F-stat, p-value).
    """
    n = len(y)
    ones = np.ones((n, 1))
    return partial_f_scratch(np.hstack([ones, X_regressors]), ones, y)


def wald_late_scratch(Y, D, Z):
    """
    Wald estimator of the LATE.

    Args:
        Y, D, Z (np.ndarray): outcome, treatment, binary instrument.

    Returns:
        tuple: (LATE, ITT_Y, ITT_D).
    """
    itt_y = Y[Z == 1].mean() - Y[Z == 0].mean()
    itt_d = D[Z == 1].mean() - D[Z == 0].mean()
    return itt_y / itt_d, itt_y, itt_d


def wald_late_boot_scratch(Y, D, Z, reps=500, seed=12345):
    """
    Bootstrap standard error for the Wald LATE estimator.

    Args:
        Y, D, Z (np.ndarray): outcome, treatment, instrument.
        reps (int): bootstrap replications.
        seed (int): RNG seed.

    Returns:
        float: bootstrap SE.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    boot_late = []
    for _ in range(reps):
        idx = rng.choice(n, size=n, replace=True)
        try:
            late_b, _, _ = wald_late_scratch(Y[idx], D[idx], Z[idx])
            if np.isfinite(late_b):
                boot_late.append(late_b)
        except Exception:
            continue
    return float(np.std(boot_late))


def two_sls_scratch(Y, D, Z, W=None):
    """
    2SLS via projection. Homoskedastic SEs.

    Args:
        Y (np.ndarray): outcome.
        D (np.ndarray): endogenous regressor.
        Z (np.ndarray): instrument.
        W (np.ndarray or None): optional exogenous covariates.

    Returns:
        tuple: (beta, SE).
    """
    n = len(Y)
    if W is None:
        Z_full = np.column_stack([np.ones(n), Z])
        X_full = np.column_stack([np.ones(n), D])
    else:
        Z_full = np.column_stack([np.ones(n), Z, W])
        X_full = np.column_stack([np.ones(n), D, W])

    # First stage: project X_full onto column space of Z_full
    PZ = Z_full @ np.linalg.pinv(Z_full.T @ Z_full) @ Z_full.T
    X_hat = PZ @ X_full

    # Second stage coefficients
    beta = np.linalg.pinv(X_hat.T @ X_full) @ (X_hat.T @ Y)

    # Residuals computed using original X, not X_hat (essential for valid SE)
    resid = Y - X_full @ beta
    sigma2 = (resid @ resid) / (n - X_full.shape[1])

    # Variance: sigma^2 * (X_hat' X_full)^{-1}
    var_b = sigma2 * np.linalg.pinv(X_hat.T @ X_full)
    se = np.sqrt(np.diag(var_b))
    return beta, se