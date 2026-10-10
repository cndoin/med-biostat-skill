# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: pymr@maxghenis
# 原路径: _upstream/pymr/src/pymr/multivariable.py
# 许可: Unlicense（公有领域）
# 说明: 多变量 MR（MVMR）：IVW 推广 + 工具变量强度诊断
# 源 sha256: ece1ba0038d3ae32
# VENDORED-FILE-END
"""Multivariable Mendelian Randomization.

Estimate direct causal effects of multiple exposures on an outcome,
controlling for the effects of other exposures.

This addresses the limitation of univariable MR when exposures are
correlated or share genetic architecture.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class MultivariableMR:
    """Multivariable Mendelian Randomization.

    Estimates the direct effect of each exposure on the outcome,
    adjusted for all other exposures.

    Example:
        >>> mvmr = MultivariableMR()
        >>> mvmr.add_exposure("BMI", bmi_betas, bmi_ses)
        >>> mvmr.add_exposure("WHR", whr_betas, whr_ses)
        >>> mvmr.set_outcome(outcome_betas, outcome_ses)
        >>> results = mvmr.fit()
    """

    def __init__(self) -> None:
        """Initialize empty MVMR model."""
        self.exposures: dict[str, dict[str, NDArray]] = {}
        self.outcome: dict[str, NDArray] | None = None
        self.snp_ids: list[str] | None = None

    def add_exposure(
        self,
        name: str,
        beta: NDArray[np.floating[Any]],
        se: NDArray[np.floating[Any]],
    ) -> "MultivariableMR":
        """Add an exposure to the model.

        Args:
            name: Name of the exposure (e.g., "BMI", "WHR")
            beta: Effect sizes of SNPs on this exposure
            se: Standard errors

        Returns:
            self for method chaining
        """
        self.exposures[name] = {"beta": np.asarray(beta), "se": np.asarray(se)}
        return self

    def set_outcome(
        self,
        beta: NDArray[np.floating[Any]],
        se: NDArray[np.floating[Any]],
        snp_ids: list[str] | None = None,
    ) -> "MultivariableMR":
        """Set the outcome GWAS data.

        Args:
            beta: Effect sizes of SNPs on outcome
            se: Standard errors
            snp_ids: Optional SNP identifiers

        Returns:
            self for method chaining
        """
        self.outcome = {"beta": np.asarray(beta), "se": np.asarray(se)}
        self.snp_ids = snp_ids
        return self

    def _validate(self) -> None:
        """Validate data before fitting."""
        if len(self.exposures) < 2:
            msg = "MVMR requires at least 2 exposures"
            raise ValueError(msg)

        if self.outcome is None:
            msg = "Outcome not set. Call set_outcome() first."
            raise ValueError(msg)

        # Check all arrays have same length
        n = len(self.outcome["beta"])
        for name, data in self.exposures.items():
            if len(data["beta"]) != n:
                msg = f"Exposure '{name}' has {len(data['beta'])} SNPs, expected {n}"
                raise ValueError(msg)

    def fit(self, method: str = "IVW") -> pd.DataFrame:
        """Fit the MVMR model.

        Args:
            method: Estimation method. Options:
                - "IVW": Inverse variance weighted (default)
                - "Egger": MR-Egger with intercept
                - "Lasso": L1-penalized for variable selection

        Returns:
            DataFrame with effect estimates for each exposure
        """
        self._validate()

        if method == "IVW":
            return self._fit_ivw()
        elif method == "Egger":
            return self._fit_egger()
        elif method == "Lasso":
            return self._fit_lasso()
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

    def _fit_ivw(self) -> pd.DataFrame:
        """Fit MVMR using IVW (weighted least squares)."""
        # Build design matrix X (exposures) and outcome vector y
        exposure_names = list(self.exposures.keys())
        n_snps = len(self.outcome["beta"])
        n_exp = len(exposure_names)

        X = np.column_stack([
            self.exposures[name]["beta"] for name in exposure_names
        ])
        y = self.outcome["beta"]

        # Weights: inverse outcome variance
        weights = 1 / self.outcome["se"] ** 2
        W = np.diag(weights)

        # Weighted least squares: (X'WX)^-1 X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y

        try:
            beta = np.linalg.solve(XtWX, XtWy)

            # Standard errors
            residuals = y - X @ beta
            sigma2 = np.sum(weights * residuals**2) / (n_snps - n_exp)
            var_beta = sigma2 * np.linalg.inv(XtWX)
            se = np.sqrt(np.diag(var_beta))
        except np.linalg.LinAlgError:
            beta = np.zeros(n_exp)
            se = np.full(n_exp, np.nan)

        # P-values
        pval = 2 * stats.norm.sf(np.abs(beta / np.maximum(se, 1e-10)))

        return pd.DataFrame({
            "exposure": exposure_names,
            "beta": beta,
            "se": se,
            "pval": pval,
            "OR": np.exp(beta),
            "OR_lci": np.exp(beta - 1.96 * se),
            "OR_uci": np.exp(beta + 1.96 * se),
            "method": "MVMR-IVW",
            "n_snps": n_snps,
        })

    def _fit_egger(self) -> pd.DataFrame:
        """Fit MVMR-Egger (with intercept for pleiotropy)."""
        exposure_names = list(self.exposures.keys())
        n_snps = len(self.outcome["beta"])
        n_exp = len(exposure_names)

        # Add intercept column
        X = np.column_stack([
            np.ones(n_snps),  # Intercept
            *[self.exposures[name]["beta"] for name in exposure_names]
        ])
        y = self.outcome["beta"]

        weights = 1 / self.outcome["se"] ** 2
        W = np.diag(weights)

        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y

        try:
            coef = np.linalg.solve(XtWX, XtWy)

            residuals = y - X @ coef
            sigma2 = np.sum(weights * residuals**2) / (n_snps - n_exp - 1)
            var_coef = sigma2 * np.linalg.inv(XtWX)
            se_coef = np.sqrt(np.diag(var_coef))
        except np.linalg.LinAlgError:
            coef = np.zeros(n_exp + 1)
            se_coef = np.full(n_exp + 1, np.nan)

        # Extract intercept and slopes
        intercept = coef[0]
        intercept_se = se_coef[0]
        beta = coef[1:]
        se = se_coef[1:]

        pval = 2 * stats.norm.sf(np.abs(beta / np.maximum(se, 1e-10)))
        intercept_pval = 2 * stats.norm.sf(np.abs(intercept / max(intercept_se, 1e-10)))

        result = pd.DataFrame({
            "exposure": exposure_names,
            "beta": beta,
            "se": se,
            "pval": pval,
            "OR": np.exp(beta),
            "OR_lci": np.exp(beta - 1.96 * se),
            "OR_uci": np.exp(beta + 1.96 * se),
            "method": "MVMR-Egger",
            "n_snps": n_snps,
        })

        # Add intercept info as attributes
        result.attrs["intercept"] = intercept
        result.attrs["intercept_se"] = intercept_se
        result.attrs["intercept_pval"] = intercept_pval

        return result

    def _fit_lasso(self, alpha: float = 0.1) -> pd.DataFrame:
        """Fit MVMR with L1 penalty for exposure selection.

        Uses coordinate descent for L1-penalized weighted least squares.
        """
        exposure_names = list(self.exposures.keys())
        n_snps = len(self.outcome["beta"])
        n_exp = len(exposure_names)

        X = np.column_stack([
            self.exposures[name]["beta"] for name in exposure_names
        ])
        y = self.outcome["beta"]

        weights = 1 / self.outcome["se"] ** 2

        # Standardize X for fair penalization
        X_scaled = X * np.sqrt(weights)[:, np.newaxis]
        y_scaled = y * np.sqrt(weights)

        # Coordinate descent
        beta = np.zeros(n_exp)
        max_iter = 1000
        tol = 1e-6

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(n_exp):
                # Partial residual
                r = y_scaled - X_scaled @ beta + X_scaled[:, j] * beta[j]

                # Soft thresholding
                z = X_scaled[:, j] @ r
                norm = np.sum(X_scaled[:, j] ** 2)

                if z > alpha:
                    beta[j] = (z - alpha) / norm
                elif z < -alpha:
                    beta[j] = (z + alpha) / norm
                else:
                    beta[j] = 0

            if np.max(np.abs(beta - beta_old)) < tol:
                break

        # Bootstrap SE (simplified)
        rng = np.random.default_rng(42)
        beta_samples = []
        for _ in range(100):
            idx = rng.choice(n_snps, size=n_snps, replace=True)
            X_boot = X_scaled[idx]
            y_boot = y_scaled[idx]

            beta_boot = np.zeros(n_exp)
            for _ in range(100):
                for j in range(n_exp):
                    r = y_boot - X_boot @ beta_boot + X_boot[:, j] * beta_boot[j]
                    z = X_boot[:, j] @ r
                    norm = np.sum(X_boot[:, j] ** 2)
                    if z > alpha:
                        beta_boot[j] = (z - alpha) / norm
                    elif z < -alpha:
                        beta_boot[j] = (z + alpha) / norm
                    else:
                        beta_boot[j] = 0
            beta_samples.append(beta_boot)

        se = np.std(beta_samples, axis=0)
        se = np.maximum(se, 1e-10)
        pval = 2 * stats.norm.sf(np.abs(beta / se))

        return pd.DataFrame({
            "exposure": exposure_names,
            "beta": beta,
            "se": se,
            "pval": pval,
            "OR": np.exp(beta),
            "OR_lci": np.exp(beta - 1.96 * se),
            "OR_uci": np.exp(beta + 1.96 * se),
            "method": "MVMR-Lasso",
            "n_snps": n_snps,
            "selected": beta != 0,
        })

    def conditional_f(self) -> pd.DataFrame:
        """Calculate conditional F-statistics for instrument strength.

        Tests whether instruments are strong for each exposure after
        adjusting for the other exposures.

        Returns:
            DataFrame with F-statistic for each exposure
        """
        self._validate()

        exposure_names = list(self.exposures.keys())
        n_snps = len(self.outcome["beta"])
        n_exp = len(exposure_names)

        results = []
        for i, name in enumerate(exposure_names):
            # Regress exposure i on all other exposures
            other_X = np.column_stack([
                self.exposures[other]["beta"]
                for j, other in enumerate(exposure_names) if j != i
            ])
            y = self.exposures[name]["beta"]

            # Get residuals
            if other_X.shape[1] > 0:
                coef = np.linalg.lstsq(other_X, y, rcond=None)[0]
                residuals = y - other_X @ coef
            else:
                residuals = y

            # F-statistic
            ss_reg = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_reg / ss_tot if ss_tot > 0 else 0

            f_stat = (r2 * (n_snps - n_exp)) / ((1 - r2) * (n_exp - 1 + 1e-10))

            results.append({
                "exposure": name,
                "conditional_F": f_stat,
                "conditional_R2": r2,
                "strong_instrument": f_stat > 10,
            })

        return pd.DataFrame(results)


def mvmr_ivw(
    exposures: dict[str, tuple[NDArray, NDArray]],
    outcome_beta: NDArray[np.floating[Any]],
    outcome_se: NDArray[np.floating[Any]],
) -> pd.DataFrame:
    """Convenience function for MVMR-IVW.

    Args:
        exposures: Dict mapping exposure names to (beta, se) tuples
        outcome_beta: Outcome effect sizes
        outcome_se: Outcome standard errors

    Returns:
        DataFrame with effect estimates for each exposure
    """
    mvmr = MultivariableMR()
    for name, (beta, se) in exposures.items():
        mvmr.add_exposure(name, beta, se)
    mvmr.set_outcome(outcome_beta, outcome_se)
    return mvmr.fit(method="IVW")
