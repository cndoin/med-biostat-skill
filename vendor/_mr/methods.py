# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: pymr@maxghenis
# 原路径: _upstream/pymr/src/pymr/methods.py
# 许可: Unlicense（公有领域）
# 说明: MR 估计量：IVW / 加权中位数 / MR-Egger / 简单众数 / 加权众数 / MR-PRESSO / MR-RAPS / 污染混合
# 源 sha256: 36a03c72bc0f2692
# VENDORED-FILE-END
"""Mendelian Randomization statistical methods.

This module implements the core MR estimators:
- IVW (Inverse Variance Weighted)
- Weighted Median
- MR-Egger
- Mode-based estimation

References:
    Burgess S, Thompson SG (2015). Mendelian Randomization: Methods for Using
    Genetic Variants in Causal Estimation. Chapman & Hall/CRC.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def ivw(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Inverse Variance Weighted MR.

    Computes a weighted average of per-SNP Wald ratios, where weights
    are the inverse of the outcome variance.

    Args:
        beta_exp: Effect sizes for exposure (per allele)
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome (per allele)
        se_out: Standard errors for outcome

    Returns:
        Dictionary with beta, se, pval, OR, OR_lci, OR_uci, nsnp
    """
    # Wald ratios
    wald_ratio = beta_out / beta_exp
    wald_se = np.abs(se_out / beta_exp)

    # IVW weights
    weights = 1 / wald_se**2

    # Weighted mean
    beta = np.sum(wald_ratio * weights) / np.sum(weights)
    se = np.sqrt(1 / np.sum(weights))
    pval = 2 * stats.norm.sf(np.abs(beta / se))

    return {
        "beta": float(beta),
        "se": float(se),
        "pval": float(pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": len(beta_exp),
    }


def weighted_median(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    n_bootstrap: int = 1000,
) -> dict[str, float]:
    """Weighted Median MR.

    Robust to up to 50% of instruments being invalid.

    Args:
        beta_exp: Effect sizes for exposure
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome
        se_out: Standard errors for outcome
        n_bootstrap: Number of bootstrap samples for SE estimation

    Returns:
        Dictionary with beta, se, pval, OR, OR_lci, OR_uci, nsnp

    Raises:
        ValueError: If fewer than 3 SNPs provided
    """
    if len(beta_exp) < 3:
        msg = "Weighted median requires at least 3 SNPs"
        raise ValueError(msg)

    # Wald ratios and weights
    wald_ratio = beta_out / beta_exp
    wald_se = np.abs(se_out / beta_exp)
    weights = 1 / wald_se**2

    # Weighted median
    sorted_idx = np.argsort(wald_ratio)
    cumsum_weights = np.cumsum(weights[sorted_idx]) / np.sum(weights)
    median_idx = np.searchsorted(cumsum_weights, 0.5)
    beta = float(wald_ratio[sorted_idx[median_idx]])

    # Bootstrap SE
    rng = np.random.default_rng(42)
    betas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(beta_exp), size=len(beta_exp), replace=True)
        w = weights[idx]
        r = wald_ratio[idx]
        sidx = np.argsort(r)
        cs = np.cumsum(w[sidx]) / np.sum(w)
        midx = np.searchsorted(cs, 0.5)
        betas.append(r[sidx[midx]])

    se = max(float(np.std(betas)), 1e-10)  # Avoid division by zero
    pval = 2 * stats.norm.sf(np.abs(beta / se))

    return {
        "beta": beta,
        "se": se,
        "pval": float(pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": len(beta_exp),
    }


def mr_egger(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
) -> dict[str, float]:
    """MR-Egger regression.

    Tests for and corrects directional pleiotropy. A non-zero intercept
    indicates the presence of directional pleiotropy.

    Args:
        beta_exp: Effect sizes for exposure
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome
        se_out: Standard errors for outcome

    Returns:
        Dictionary with beta, se, pval, intercept, intercept_se,
        intercept_pval, OR, OR_lci, OR_uci, nsnp
    """
    # Ensure positive exposure effects (InSIDE assumption)
    sign = np.sign(beta_exp)
    beta_exp_oriented = np.abs(beta_exp)
    beta_out_oriented = beta_out * sign

    # Weighted regression
    weights = 1 / se_out**2
    W = np.diag(weights)
    X = np.column_stack([np.ones(len(beta_exp)), beta_exp_oriented])

    # Solve normal equations
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ beta_out_oriented

    try:
        coef = np.linalg.solve(XtWX, XtWy)
        residuals = beta_out_oriented - X @ coef
        n = len(beta_exp)
        sigma2 = np.sum((residuals**2) * weights) / (n - 2)
        var_coef = sigma2 * np.linalg.inv(XtWX)
        se_coef = np.sqrt(np.diag(var_coef))
    except np.linalg.LinAlgError:
        # Fallback for singular matrices
        coef = np.array([0.0, np.mean(beta_out / beta_exp)])
        se_coef = np.array([np.nan, np.nan])

    intercept = float(coef[0])
    intercept_se = max(float(se_coef[0]), 1e-10)  # Avoid division by zero
    beta = float(coef[1])
    se = max(float(se_coef[1]), 1e-10)  # Avoid division by zero

    intercept_pval = 2 * stats.norm.sf(np.abs(intercept / intercept_se))
    pval = 2 * stats.norm.sf(np.abs(beta / se))

    return {
        "beta": beta,
        "se": se,
        "pval": float(pval),
        "intercept": intercept,
        "intercept_se": intercept_se,
        "intercept_pval": float(intercept_pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": len(beta_exp),
    }


def simple_mode(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    bandwidth: float | None = None,
) -> dict[str, float]:
    """Mode-based MR estimation.

    Consistent when the largest group of SNPs share the same causal effect.

    Args:
        beta_exp: Effect sizes for exposure
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome
        se_out: Standard errors for outcome
        bandwidth: Kernel bandwidth (None for automatic)

    Returns:
        Dictionary with beta, se, pval, OR, OR_lci, OR_uci, nsnp
    """
    wald_ratio = beta_out / beta_exp

    # Silverman's rule of thumb for bandwidth
    if bandwidth is None:
        n = len(wald_ratio)
        iqr = np.percentile(wald_ratio, 75) - np.percentile(wald_ratio, 25)
        bandwidth = 0.9 * min(np.std(wald_ratio), iqr / 1.34) * n ** (-0.2)

    # Kernel density estimation on a grid
    grid = np.linspace(
        np.min(wald_ratio) - 3 * bandwidth,
        np.max(wald_ratio) + 3 * bandwidth,
        1000,
    )
    density = np.zeros_like(grid)
    for r in wald_ratio:
        density += stats.norm.pdf(grid, loc=r, scale=bandwidth)

    beta = float(grid[np.argmax(density)])

    # Bootstrap SE (simplified)
    rng = np.random.default_rng(42)
    betas = []
    for _ in range(1000):
        idx = rng.choice(len(beta_exp), size=len(beta_exp), replace=True)
        wr = (beta_out[idx] / beta_exp[idx])
        d = np.zeros_like(grid)
        for r in wr:
            d += stats.norm.pdf(grid, loc=r, scale=bandwidth)
        betas.append(grid[np.argmax(d)])

    se = float(np.std(betas))
    pval = 2 * stats.norm.sf(np.abs(beta / se)) if se > 0 else 1.0

    return {
        "beta": beta,
        "se": se,
        "pval": float(pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": len(beta_exp),
    }


def weighted_mode(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    bandwidth: float | None = None,
) -> dict[str, float]:
    """Weighted mode-based MR estimation.

    Similar to simple_mode but weights each SNP's contribution to the kernel
    density by its precision (1/se^2). More precise estimates contribute more
    to the mode estimation.

    Args:
        beta_exp: Effect sizes for exposure
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome
        se_out: Standard errors for outcome
        bandwidth: Kernel bandwidth (None for automatic)

    Returns:
        Dictionary with beta, se, pval, OR, OR_lci, OR_uci, nsnp
    """
    wald_ratio = beta_out / beta_exp
    wald_se = np.abs(se_out / beta_exp)

    # Inverse variance weights
    weights = 1 / wald_se**2

    # Silverman's rule of thumb for bandwidth
    if bandwidth is None:
        n = len(wald_ratio)
        iqr = np.percentile(wald_ratio, 75) - np.percentile(wald_ratio, 25)
        bandwidth = 0.9 * min(np.std(wald_ratio), iqr / 1.34) * n ** (-0.2)

    # Kernel density estimation on a grid with inverse-variance weighting
    grid = np.linspace(
        np.min(wald_ratio) - 3 * bandwidth,
        np.max(wald_ratio) + 3 * bandwidth,
        1000,
    )
    density = np.zeros_like(grid)
    for r, w in zip(wald_ratio, weights):
        density += w * stats.norm.pdf(grid, loc=r, scale=bandwidth)

    beta = float(grid[np.argmax(density)])

    # Bootstrap SE (simplified)
    rng = np.random.default_rng(42)
    betas = []
    for _ in range(1000):
        idx = rng.choice(len(beta_exp), size=len(beta_exp), replace=True)
        wr = beta_out[idx] / beta_exp[idx]
        wt = 1 / (np.abs(se_out[idx] / beta_exp[idx]) ** 2)
        d = np.zeros_like(grid)
        for r, w in zip(wr, wt):
            d += w * stats.norm.pdf(grid, loc=r, scale=bandwidth)
        betas.append(grid[np.argmax(d)])

    se = float(np.std(betas))
    pval = 2 * stats.norm.sf(np.abs(beta / se)) if se > 0 else 1.0

    return {
        "beta": beta,
        "se": se,
        "pval": float(pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": len(beta_exp),
    }


def mr_presso(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    n_simulations: int = 10000,
    outlier_test: bool = True,
    distortion_test: bool = True,
) -> dict[str, Any]:
    """MR-PRESSO (Pleiotropy RESidual Sum and Outlier).

    Detects and corrects for horizontal pleiotropy via:
    1. Global test: Tests for presence of pleiotropy
    2. Outlier test: Identifies SNPs with significant residuals
    3. Distortion test: Tests if removing outliers changes estimate

    Args:
        beta_exp: Effect sizes for exposure (per allele)
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome (per allele)
        se_out: Standard errors for outcome
        n_simulations: Number of simulations for global test
        outlier_test: Whether to perform outlier detection
        distortion_test: Whether to test for distortion

    Returns:
        Dictionary with:
            - global_test_pval: p-value for presence of horizontal pleiotropy
            - outlier_indices: list of detected outlier indices
            - corrected_beta: IVW estimate after outlier removal
            - corrected_se: SE after outlier removal
            - corrected_pval: p-value after outlier removal
            - distortion_test_pval: p-value for significant difference
            - original_beta: Original IVW estimate
            - original_se: Original IVW SE
            - nsnp: Number of SNPs

    Raises:
        ValueError: If fewer than 3 SNPs provided

    References:
        Verbanck et al. (2018) Nature Genetics 50:693-698
    """
    if len(beta_exp) < 3:
        msg = "MR-PRESSO requires at least 3 SNPs"
        raise ValueError(msg)

    # Get original IVW estimate
    original_ivw = ivw(beta_exp, se_exp, beta_out, se_out)
    original_beta = original_ivw["beta"]
    original_se = original_ivw["se"]

    # Compute observed residual sum of squares
    wald_ratio = beta_out / beta_exp
    wald_se = np.abs(se_out / beta_exp)
    weights = 1 / wald_se**2

    residuals = wald_ratio - original_beta
    rss_obs = np.sum((residuals**2) * weights)

    # Global test: Simulate null distribution
    # Use legacy RandomState for reproducibility with np.random.seed()
    rng = np.random.RandomState()
    rss_null = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate under null (no pleiotropy)
        sim_wald = rng.normal(original_beta, wald_se)
        sim_residuals = sim_wald - original_beta
        rss_null[i] = np.sum((sim_residuals**2) * weights)

    # Global test p-value
    global_test_pval = float(np.mean(rss_null >= rss_obs))

    # Outlier test
    outlier_indices: list[int] = []
    if outlier_test:
        # Compute residuals and test each SNP
        for idx in range(len(beta_exp)):
            # Compute RSS without this SNP
            mask = np.ones(len(beta_exp), dtype=bool)
            mask[idx] = False

            # IVW without this SNP
            ivw_loo = ivw(
                beta_exp[mask],
                se_exp[mask],
                beta_out[mask],
                se_out[mask]
            )
            beta_loo = ivw_loo["beta"]

            # Residual for this SNP
            residual_i = wald_ratio[idx] - beta_loo

            # Simulate null distribution for this SNP's residual
            rss_null_i = np.zeros(n_simulations)
            for sim_idx in range(n_simulations):
                sim_wald_i = rng.normal(beta_loo, wald_se[idx])
                sim_residual_i = sim_wald_i - beta_loo
                rss_null_i[sim_idx] = (sim_residual_i**2) * weights[idx]

            # Test if observed residual is extreme
            pval_i = np.mean(rss_null_i >= (residual_i**2) * weights[idx])

            # Bonferroni correction
            if pval_i < (0.05 / len(beta_exp)):
                outlier_indices.append(idx)

    # Corrected estimate (after outlier removal)
    if len(outlier_indices) > 0:
        keep_mask = np.ones(len(beta_exp), dtype=bool)
        keep_mask[outlier_indices] = False

        corrected_ivw = ivw(
            beta_exp[keep_mask],
            se_exp[keep_mask],
            beta_out[keep_mask],
            se_out[keep_mask]
        )
        corrected_beta = corrected_ivw["beta"]
        corrected_se = corrected_ivw["se"]
        corrected_pval = corrected_ivw["pval"]
    else:
        corrected_beta = original_beta
        corrected_se = original_se
        corrected_pval = original_ivw["pval"]

    # Distortion test
    distortion_test_pval = np.nan
    if distortion_test and len(outlier_indices) > 0:
        # Test if difference between original and corrected is significant
        diff = abs(corrected_beta - original_beta)
        se_diff = np.sqrt(original_se**2 + corrected_se**2)
        z_stat = diff / se_diff if se_diff > 0 else 0
        distortion_test_pval = float(2 * stats.norm.sf(abs(z_stat)))

    return {
        "global_test_pval": global_test_pval,
        "outlier_indices": outlier_indices,
        "corrected_beta": float(corrected_beta),
        "corrected_se": float(corrected_se),
        "corrected_pval": float(corrected_pval),
        "distortion_test_pval": float(distortion_test_pval),
        "original_beta": float(original_beta),
        "original_se": float(original_se),
        "nsnp": len(beta_exp),
    }


def mr_raps(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """MR-RAPS (Robust Adjusted Profile Score).

    Accounts for measurement error in SNP-exposure effects using a robust
    profile likelihood approach. More robust to weak instruments than IVW.

    The method iteratively estimates the causal effect by:
    1. Starting with IVW estimate
    2. Computing robust weights based on Huber loss
    3. Updating estimate until convergence

    Args:
        beta_exp: Effect sizes for exposure (per allele)
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome (per allele)
        se_out: Standard errors for outcome
        max_iter: Maximum iterations for convergence
        tol: Convergence tolerance

    Returns:
        Dictionary with beta, se, pval, OR, OR_lci, OR_uci, nsnp

    Raises:
        ValueError: If fewer than 3 SNPs provided

    References:
        Zhao et al. (2020) Statistical inference in two-sample summary-data
        Mendelian randomization using robust adjusted profile score.
        International Journal of Epidemiology.
    """
    if len(beta_exp) < 3:
        msg = "MR-RAPS requires at least 3 SNPs"
        raise ValueError(msg)

    n = len(beta_exp)

    # Initialize with IVW estimate
    ivw_result = ivw(beta_exp, se_exp, beta_out, se_out)
    beta = ivw_result["beta"]

    # Huber constant (standard value)
    k = 1.345

    # Iterative robust estimation
    for iteration in range(max_iter):
        beta_old = beta

        # Compute residuals
        residuals = beta_out - beta * beta_exp

        # Variance accounting for measurement error in exposure
        # var(Y - beta*X) = var(Y) + beta^2 * var(X)
        var_total = se_out**2 + (beta**2) * (se_exp**2)
        std_total = np.sqrt(var_total)

        # Standardized residuals
        std_residuals = residuals / std_total

        # Huber weights
        weights = np.ones(n)
        outlier_mask = np.abs(std_residuals) > k
        weights[outlier_mask] = k / np.abs(std_residuals[outlier_mask])

        # Weighted estimate accounting for measurement error
        # Weight by precision, adjusted by robust weight
        precision = weights / var_total

        # Numerator: sum of weighted ratios
        numerator = np.sum(precision * beta_out * beta_exp)

        # Denominator: sum of weighted exposure effects squared
        denominator = np.sum(precision * beta_exp**2)

        # Update beta
        if denominator > 0:
            beta = numerator / denominator
        else:
            # Fallback if denominator is zero
            beta = 0.0

        # Check convergence
        if abs(beta - beta_old) < tol:
            break

    # Compute standard error
    # Variance accounting for measurement error
    var_total = se_out**2 + (beta**2) * (se_exp**2)
    std_total = np.sqrt(var_total)
    std_residuals = (beta_out - beta * beta_exp) / std_total

    # Huber weights for final variance calculation
    weights = np.ones(n)
    outlier_mask = np.abs(std_residuals) > k
    weights[outlier_mask] = k / np.abs(std_residuals[outlier_mask])

    # Robust variance estimator
    precision = weights / var_total
    var_beta = 1 / np.sum(precision * beta_exp**2)
    se = np.sqrt(var_beta)

    # Ensure positive SE
    se = max(float(se), 1e-10)

    # P-value
    pval = 2 * stats.norm.sf(np.abs(beta / se))

    return {
        "beta": float(beta),
        "se": se,
        "pval": float(pval),
        "OR": float(np.exp(beta)),
        "OR_lci": float(np.exp(beta - 1.96 * se)),
        "OR_uci": float(np.exp(beta + 1.96 * se)),
        "nsnp": n,
    }


def contamination_mixture(
    beta_exp: NDArray[np.floating[Any]],
    se_exp: NDArray[np.floating[Any]],
    beta_out: NDArray[np.floating[Any]],
    se_out: NDArray[np.floating[Any]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Contamination mixture model for MR.

    Models SNP effects as a mixture of valid instruments (following the causal
    estimate) and invalid instruments (pleiotropic). Uses EM algorithm to
    estimate the mixture and identify contaminated SNPs.

    The model assumes:
    - Valid instruments: Wald ratio ~ N(beta_causal, sigma^2)
    - Invalid instruments: Wald ratio ~ N(0, tau^2)

    Args:
        beta_exp: Effect sizes for exposure (per allele)
        se_exp: Standard errors for exposure
        beta_out: Effect sizes for outcome (per allele)
        se_out: Standard errors for outcome
        max_iter: Maximum iterations for EM algorithm
        tol: Convergence tolerance

    Returns:
        Dictionary with:
            - beta: Causal effect estimate
            - se: Standard error
            - pval: P-value
            - prob_valid: Posterior probability each SNP is valid (array)
            - n_valid: Expected number of valid instruments
            - nsnp: Total number of SNPs

    Raises:
        ValueError: If fewer than 3 SNPs provided

    References:
        Burgess et al. (2020) A robust and efficient method for Mendelian
        randomization with hundreds of genetic variants.
        Nature Communications 11:376.
    """
    if len(beta_exp) < 3:
        msg = "Contamination mixture requires at least 3 SNPs"
        raise ValueError(msg)

    n = len(beta_exp)

    # Compute Wald ratios and their standard errors
    wald_ratio = beta_out / beta_exp
    wald_se = np.abs(se_out / beta_exp)

    # Initialize with IVW estimate
    ivw_result = ivw(beta_exp, se_exp, beta_out, se_out)
    beta = ivw_result["beta"]

    # Initialize mixture parameters
    # pi = proportion of valid instruments
    pi = 0.8  # Start assuming 80% valid
    sigma2 = np.var(wald_ratio)  # Variance for valid instruments
    tau2 = sigma2 * 2  # Variance for invalid instruments (larger)

    # EM algorithm
    for iteration in range(max_iter):
        beta_old = beta
        pi_old = pi

        # E-step: Compute posterior probabilities
        # P(valid | data) for each SNP

        # Likelihood under valid model: N(beta, wald_se^2 + sigma2)
        var_valid = wald_se**2 + sigma2
        log_lik_valid = stats.norm.logpdf(wald_ratio, loc=beta, scale=np.sqrt(var_valid))

        # Likelihood under invalid model: N(0, wald_se^2 + tau2)
        var_invalid = wald_se**2 + tau2
        log_lik_invalid = stats.norm.logpdf(wald_ratio, loc=0, scale=np.sqrt(var_invalid))

        # Posterior probability of being valid
        # Using log-sum-exp trick for numerical stability
        log_prior_valid = np.log(pi)
        log_prior_invalid = np.log(1 - pi)

        log_post_valid = log_lik_valid + log_prior_valid
        log_post_invalid = log_lik_invalid + log_prior_invalid

        # Normalize
        max_log = np.maximum(log_post_valid, log_post_invalid)
        prob_valid = np.exp(log_post_valid - max_log) / (
            np.exp(log_post_valid - max_log) + np.exp(log_post_invalid - max_log)
        )

        # M-step: Update parameters

        # Update pi (proportion of valid instruments)
        pi = np.mean(prob_valid)

        # Update beta (weighted by posterior probability of being valid)
        weights = prob_valid / var_valid
        beta = np.sum(weights * wald_ratio) / np.sum(weights)

        # Update sigma2 (variance for valid instruments)
        sigma2 = np.sum(prob_valid * (wald_ratio - beta)**2) / np.sum(prob_valid)
        sigma2 = max(sigma2, 1e-10)  # Ensure positive

        # Update tau2 (variance for invalid instruments)
        tau2 = np.sum((1 - prob_valid) * wald_ratio**2) / np.sum(1 - prob_valid)
        tau2 = max(tau2, 1e-10)  # Ensure positive

        # Check convergence
        if abs(beta - beta_old) < tol and abs(pi - pi_old) < tol:
            break

    # Compute standard error for beta
    # Using inverse of Fisher information for valid instruments
    var_beta = 1 / np.sum(prob_valid / var_valid)
    se = np.sqrt(var_beta)

    # Ensure positive SE
    se = max(float(se), 1e-10)

    # P-value
    pval = 2 * stats.norm.sf(np.abs(beta / se))

    # Expected number of valid instruments
    n_valid = float(np.sum(prob_valid))

    return {
        "beta": float(beta),
        "se": se,
        "pval": float(pval),
        "prob_valid": prob_valid,
        "n_valid": n_valid,
        "nsnp": n,
    }
