# """
# Gene variance modeling with simplified scanpy integration.

# This module provides classes and functions for modeling gene variance
# in single-cell RNA sequencing data, decomposing it into technical and
# biological components, using scanpy as the foundation.
# """

# from enum import Enum
# from functools import partial
# from typing import Any, Callable, Dict, List, Literal, Optional, Union

# import numpy as np
# import pandas as pd

# # Required scanpy imports
# import scanpy as sc
# from anndata import AnnData
# from scanpy.preprocessing._utils import _get_mean_var
# from scipy import linalg, optimize, sparse, stats


# class ModelFlavor(Enum):
#     """Enum for different model flavors."""

#     STANDARD = "standard"
#     POISSON = "poisson"


# class TrendFitMethod(Enum):
#     """Enum for different trend fitting methods."""

#     PARAMETRIC = "parametric"
#     LOWESS = "lowess"
#     BOTH = "both"


# class TrendFitter:
#     """Class for fitting mean-variance trends."""

#     do_parametric: bool = True
#     do_lowess: bool = True
#     use_density_weights: bool = True
#     span: float = 0.3

#     def __init__(
#         self,
#         flavor: Literal["parametric", "lowess", "both"] = "both",
#         use_density_weights: bool = True,
#         span: float = 0.3,
#     ):
#         assert flavor in ["parametric", "lowess", "both"], (
#             f"invalid 'flavor' provided: {flavor}."
#         )
#         assert (span > 0.0) and (span <= 1.0), f"'span' must be between 0 and 1: {span}"
#         if flavor == "parametric":
#             self.do_lowess = False
#         elif flavor == "lowess":
#             self.do_parametric = False
#         self.use_density_weights = use_density_weights
#         self.span = span

#     @staticmethod
#     def correct_logged_expectation(
#         x: np.ndarray, y: np.ndarray, w: np.ndarray, func: Callable
#     ) -> Dict[str, Any]:
#         """
#         Adjust for scale shift due to fitting to log-values.

#         Args:
#             x: x values
#             y: y values
#             w: Weights
#             func: Function to generate trend

#         Returns:
#             Dict with trend function and standard deviation
#         """
#         from ._loess_fit import weighted_median

#         # Calculate leftovers
#         func_vals = [func(x) for x in np.asarray(x)]
#         leftovers = y / func_vals

#         # Calculate weighted median
#         med = weighted_median(leftovers, w, na_rm=True)

#         return dict(
#             trend_func=(lambda x: func(x) * med),
#             std_dev=(
#                 weighted_median(np.abs(leftovers / med - 1), w, na_rm=True) * 1.4826
#             ),
#         )

#     @staticmethod
#     def get_init_params(
#         vars: np.ndarray,
#         means: np.ndarray,
#         left_n: int = 100,
#         left_prop: float = 0.1,
#         grid_length: int = 10,
#         b_grid_range: float = 5,
#         n_grid_max: float = 10,
#     ) -> Dict[str, float]:
#         """
#         Get starting parameters for non-linear curve fitting.

#         Args:
#             vars: Variances
#             means: Means
#             left_n: Number of points to use from left
#             left_prop: Proportion of points to use from left
#             grid_length: Number of grid points
#             b_grid_range: Range for B parameter grid
#             n_grid_max: Maximum value for n parameter grid

#         Returns:
#             Dict with starting parameters
#         """

#         # Sort by means
#         n = len(vars)
#         sorted_idx = np.argsort(means)

#         # Estimate gradient from left
#         _left_n = min(left_n, int(n * left_prop))
#         keep_idx = sorted_idx[: max(1, _left_n)]
#         _vars = vars[keep_idx]
#         _means = means[keep_idx]

#         # Linear regression through origin
#         slope = np.sum(_means * _vars) / np.sum(_means**2)

#         # Grid search for remaining parameters
#         b_grid, n_grid = np.meshgrid(
#             np.exp(np.linspace(-b_grid_range, b_grid_range, grid_length)),
#             np.exp(np.linspace(0, n_grid_max, grid_length)),
#         )
#         b_flat = b_grid.flatten()
#         n_flat = n_grid.flatten()

#         # Evaluate sum of squares for each parameter combination
#         best_ss = np.inf
#         best_idx = 0
#         for i in range(b_flat.shape[0]):
#             _b = b_flat[i]
#             _n = n_flat[i]
#             pred = (slope * _b * means) / (_b + (means**_n))
#             resd = vars - pred
#             _ss = np.dot(resd, resd)
#             if _ss < best_ss:
#                 best_ss = _ss
#                 best_idx = i

#         return dict(
#             n=max(1e-8, n_flat[best_idx] - 1),
#             b=b_flat[best_idx],
#             a=b_flat[best_idx] * slope,
#         )

#     @staticmethod
#     def _nls_model(x: float, a: float, b: float, n: float):
#         # Define the model function: y = (a*x)/(x^(1+n) + b)
#         return (a * x) / (b + (x ** (1 + n)))

#     @staticmethod
#     def _default_param(x: float, y: float):
#         return min(1, (x / y))

#     @staticmethod
#     def _lowess_unscale(x: float, loess_func: Callable, param_func: Callable):
#         return np.exp(loess_func(x)) * param_func(x)

#     def fit_trend_var(
#         self,
#         means: np.ndarray,
#         vars: np.ndarray,
#         **kwargs,
#     ) -> Dict[str, Any]:
#         from ._loess_fit import inverse_density_weights, weighted_lowess

#         assert len(vars) >= 2, "need at least 2 points for fitting."
#         w = (
#             inverse_density_weights(means, bw_method=1.0)
#             if self.use_density_weights
#             else None
#         )

#         # Default parametric trend is a straight line from 0 to min(m)
#         to_fit = np.log(vars)
#         left_edge = np.min(means)
#         _param_func = partial(TrendFitter._default_param, y=left_edge)

#         # Fit parametric curve if requested
#         if self.do_parametric:
#             assert len(vars) >= 4, (
#                 "need at least 4 points for non-linear curve fitting."
#             )
#             # Get initial parameters
#             model_params = TrendFitter.get_init_params(vars, means)
#             try:
#                 opt_params, _ = optimize.curve_fit(
#                     TrendFitter._nls_model,
#                     means,
#                     vars,
#                     p0=[model_params["a"], model_params["b"], model_params["n"]],
#                     sigma=(None if w is None else (1 / np.sqrt(w))),
#                     method="trf",
#                     bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
#                     ftol=1e-8,
#                     xtol=1e-8,
#                     gtol=1e-8,
#                     max_nfev=500,
#                 )
#                 _param_func = partial(
#                     TrendFitter._nls_model,
#                     a=opt_params[0],
#                     b=opt_params[1],
#                     n=opt_params[2],
#                 )
#             except RuntimeError:
#                 # Use initial estimates if fitting fails
#                 _param_func = partial(
#                     TrendFitter._nls_model,
#                     a=model_params["a"],
#                     b=model_params["b"],
#                     n=(model_params["n"] + 1),
#                 )
#             # Update to_fit with residuals
#             fitted_values = np.array([_param_func(x) for x in means])
#             to_fit = to_fit - np.log(fitted_values)

#         _unscaled_func = _param_func
#         if self.do_lowess:
#             from scipy.interpolate import PchipInterpolator

#             lfit = weighted_lowess(means, to_fit, weights=w, span=self.span, **kwargs)
#             _unscaled_func = partial(
#                 TrendFitter._lowess_unscale,
#                 loess_func=PchipInterpolator(
#                     x=means, y=lfit["fitted"], extrapolate=False
#                 ),
#                 param_func=_param_func,
#             )

#         # Adjust for scale shift
#         return TrendFitter.correct_logged_expectation(means, vars, w, _unscaled_func)


# def _scran_model_gene_var(
#     X,
#     min_mean: float = 0.1,
#     flavor: Literal["parametric", "lowess", "both"] = "both",
#     use_density_weights: bool = True,
#     span: float = 0.3,
#     **kwargs,
# ) -> Dict[str, pd.Series]:
#     means, vars = _get_mean_var(X)

#     # Filter out low-abundance genes
#     valid_idx = ~np.isnan(vars) & (vars > 1e-8) & (means >= min_mean)

#     _vars = vars[valid_idx]
#     tfit = TrendFitMethod(
#         flavor=flavor, use_density_weights=use_density_weights, span=span
#     )
#     fit = tfit.fit_trend_var(means=means[valid_idx], vars=_vars, **kwargs)

#     _var_norm = np.zeros_like(vars)
#     _var_norm[valid_idx] = _vars - np.array(
#         [fit["trend_func"](x) for x in means[valid_idx]]
#     )
#     return {
#         "means": means,
#         "variances": vars,
#         "variances_norm": _var_norm,
#     }


# def _scran_log_normalize(X, size_factors, target_sum: Optional[float] = None):
#     _target_sum = np.mean(size_factors) if target_sum is None else target_sum
#     return sc.pp.log1p(
#         sc.pp._normalization._normalize_data(X, counts=size_factors, after=_target_sum),
#         base=2,
#     )


# def _get_sim_mean_vars(
#     mean_llim: float,
#     mean_ulim: float,
#     size_factors,
#     target_sum: float,
#     npts: int = 1000,
#     dispersion: float = 0,
#     random_state: int = 0,
# ):
#     # range from log --> seq --> exp
#     # mean var log norm
#     pts = np.exp2(np.linspace(np.log2(mean_llim), np.log2(mean_ulim), npts))
#     _X = np.outer((size_factors / target_sum), pts)

#     rng = np.random.default_rng(seed=random_state)
#     if dispersion == 0:
#         sim_X = rng.poisson(lam=_X)
#     else:
#         size = 1.0 / dispersion
#         prob = size / (size + _X)
#         sim_X = rng.negative_binomial(n=size, p=prob)

#     return _get_mean_var(
#         _scran_log_normalize(sim_X, size_factors=size_factors, target_sum=target_sum)
#     )


# def _scran_model_gene_var_poisson(
#     X,  # must be counts
#     size_factors,
#     npts: int = 1000,
#     dispersion: float = 0,
#     min_mean: float = 0.1,
#     flavor: Literal["parametric", "lowess", "both"] = "both",
#     use_density_weights: bool = True,
#     span: float = 0.3,
#     **kwargs,
# ):
#     _target_sum = np.mean(size_factors)
#     _X = _scran_log_normalize(X, size_factors=size_factors, target_sum=_target_sum)
#     means, vars = _get_mean_var(_X)
#     valid_idx = means > 0.0
#     _means = means[valid_idx]

#     mean_lim = [np.nanmin(_means), np.nanmax(_means)]
#     frac_mean_lim = np.exp2(mean_lim) - 1
#     sim_means, sim_vars = _get_sim_mean_vars(
#         frac_mean_lim[0],
#         frac_mean_lim[1],
#         norm_factors=(size_factors / _target_sum),
#         npts=npts,
#         dispersion=dispersion,
#     )

#     # Filter out low-abundance genes
#     sim_valid_idx = ~np.isnan(sim_vars) & (sim_vars > 1e-8) & (sim_means >= min_mean)
#     tfit = TrendFitMethod(
#         flavor=flavor, use_density_weights=use_density_weights, span=span
#     )
#     fit = tfit.fit_trend_var(
#         means=sim_means[sim_valid_idx], vars=sim_vars[sim_valid_idx], **kwargs
#     )

#     _var_norm = np.zeros_like(vars)
#     _var_norm[sim_valid_idx] = _vars - np.array(
#         [fit["trend_func"](x) for x in means[sim_valid_idx]]
#     )
#     return {
#         "means": means,
#         "variances": vars,
#         "variances_norm": _var_norm,
#     }

#     _vars = vars[valid_idx]
#     tfit = TrendFitMethod(
#         flavor=flavor, use_density_weights=use_density_weights, span=span
#     )
#     fit = tfit.fit_trend_var(means=means[sim_means], vars=sim_vars, **kwargs)

#     _var_norm = np.zeros_like(vars)
#     _var_norm[sim_valid_idx] = _vars - fit["trend"]
#     return {
#         "means": means,
#         "variances": vars,
#         "variances_norm": _var_norm,
#     }


# class BaseVarianceModel:
#     """Base class for gene variance models."""

#     def __init__(
#         self,
#         min_mean: float = 0.1,
#         flavor: Literal["parametric", "lowess", "both"] = "both",
#         use_density_weights: bool = True,
#         span: float = 0.3,
#     ):
#         """
#         Initialize the model.

#         Args:
#             min_mean: Minimum mean to use for trend fitting
#             trend_method: Method for trend fitting
#             span: Span parameter for LOWESS
#             use_density_weights: Whether to use inverse density weights
#         """
#         self.min_mean = min_mean
#         self.tfit = TrendFitter(flavor=flavor)
#         self.use_density_weights = use_density_weights
#         self.span = span
#         self.results_ = None

#     def fit(
#         self,
#         X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
#         block: Optional[np.ndarray] = None,
#         design: Optional[np.ndarray] = None,
#         subset_row: Optional[np.ndarray] = None,
#         subset_fit: Optional[np.ndarray] = None,
#         **kwargs,
#     ) -> "BaseVarianceModel":
#         """
#         Fit the model.

#         Args:
#             X: Expression matrix (genes x cells)
#             block: Blocking factor
#             design: Design matrix
#             subset_row: Rows to use
#             subset_fit: Rows to use for fitting
#             **kwargs: Additional arguments

#         Returns:
#             Self
#         """
#         raise NotImplementedError("Subclasses must implement this method")

#     def _compute_stats(
#         self,
#         X: Union[np.ndarray, sparse.spmatrix],
#         block: Optional[np.ndarray] = None,
#         design: Optional[np.ndarray] = None,
#         subset_row: Optional[np.ndarray] = None,
#         **kwargs,
#     ) -> Dict[str, Any]:
#         """
#         Compute mean and variance statistics using scanpy's functions.

#         Args:
#             X: Expression matrix (genes x cells)
#             block: Blocking factor
#             design: Design matrix
#             subset_row: Rows to use
#             **kwargs: Additional arguments

#         Returns:
#             Dict with means, variances, and cell counts
#         """
#         # Subset rows if needed
#         if subset_row is not None:
#             if isinstance(X, pd.DataFrame):
#                 X = X.iloc[subset_row]
#             else:
#                 if isinstance(subset_row, (list, np.ndarray)):
#                     X = X[subset_row]
#                 else:
#                     X = X[subset_row]

#         # Handle blocking case or no blocking
#         if design is None:
#             if block is not None:
#                 # Get dimensions
#                 if sparse.issparse(X):
#                     ngenes, ncells = X.shape
#                 else:
#                     ngenes, ncells = X.shape

#                 # Check block length
#                 if ncells != len(block):
#                     raise ValueError(
#                         "Length of 'block' should match number of columns in X"
#                     )

#                 # Convert block to factor
#                 block = np.asarray(block)
#                 block_levels = np.unique(block)

#                 # Count cells per block
#                 ncells_block = np.array(
#                     [np.sum(block == level) for level in block_levels]
#                 )
#                 resid_df = ncells_block - 1

#                 if np.all(resid_df <= 0):
#                     raise ValueError(
#                         "No residual d.f. in any level of 'block' for variance estimation"
#                     )
#             else:
#                 # No blocking
#                 if sparse.issparse(X):
#                     ngenes, ncells = X.shape
#                 else:
#                     ngenes, ncells = X.shape

#                 block = np.zeros(ncells, dtype=int)
#                 block_levels = np.array([0])
#                 ncells_block = np.array([ncells])

#             # Initialize output
#             means = np.zeros((ngenes, len(block_levels)))
#             vars = np.zeros((ngenes, len(block_levels)))

#             # Process blocks
#             for i, level in enumerate(block_levels):
#                 mask = block == level

#                 if np.sum(mask) > 0:
#                     # Get data for this block
#                     if sparse.issparse(X):
#                         X_block = X[:, mask]
#                         X_block_t = X_block.T
#                     else:
#                         X_block = X[:, mask]
#                         X_block_t = X_block.T

#                     # Calculate mean and variance using scanpy's function
#                     block_means, block_vars = get_mean_var(X_block_t, axis=0)

#                     # Store results
#                     means[:, i] = block_means
#                     vars[:, i] = block_vars
#                 else:
#                     means[:, i] = np.nan
#                     vars[:, i] = np.nan

#             return {"means": means, "vars": vars, "ncells": ncells_block}

#         else:
#             # Handle design matrix case
#             if block is not None:
#                 raise ValueError("Cannot specify both 'block' and 'design'")

#             # Get dimensions
#             if sparse.issparse(X):
#                 ngenes, ncells = X.shape
#             else:
#                 ngenes, ncells = X.shape

#             # Check design dimensions
#             if design.shape[0] != ncells:
#                 raise ValueError(
#                     "Number of rows in 'design' must match number of columns in X"
#                 )

#             # Calculate residual degrees of freedom
#             resid_df = ncells - design.shape[1]
#             if resid_df <= 0:
#                 raise ValueError("No residual d.f. in 'design' for variance estimation")

#             # QR decomposition
#             q, r, p = linalg.qr(design, pivoting=True, mode="economic")
#             qr_result = {"q": q, "r": r, "pivot": p}

#             # Calculate means
#             if sparse.issparse(X):
#                 X_t = X.T
#             else:
#                 X_t = X.T

#             means, _ = get_mean_var(X_t, axis=0)

#             # Initialize output for variances
#             vars = np.zeros(ngenes)

#             # Calculate residual variances
#             for i in range(ngenes):
#                 # Get gene expression
#                 if sparse.issparse(X):
#                     y = X[i].toarray().flatten()
#                 else:
#                     y = X[i]

#                 # Fit linear model
#                 coef = np.linalg.solve(r, q.T @ y)

#                 # Calculate residuals
#                 fitted = q @ (r @ coef)
#                 residuals = y - fitted

#                 # Calculate residual variance
#                 vars[i] = np.sum(residuals**2) / resid_df

#             return {
#                 "means": means.reshape(-1, 1),
#                 "vars": vars.reshape(-1, 1),
#                 "ncells": np.array([ncells]),
#             }

#     def _decompose_variance(
#         self, x_stats: Dict[str, np.ndarray], fit_stats: Dict[str, np.ndarray]
#     ) -> List[Dict[str, Any]]:
#         """
#         Decompose variance into components.

#         Args:
#             x_stats: Statistics for the data
#             fit_stats: Statistics for the fit

#         Returns:
#             List of dicts with decomposition results
#         """
#         # Initialize output
#         collected = []

#         # Process each block
#         for i in range(x_stats["means"].shape[1]):
#             fm = fit_stats["means"][:, i]
#             fv = fit_stats["vars"][:, i]

#             # Fit trend
#             if x_stats["ncells"][i] >= 2:
#                 fit = TrendFitter.fit_trend_var(
#                     fm,
#                     fv,
#                     min_mean=self.min_mean,
#                     flavor=self.trend_method,
#                     use_density_weights=self.use_density_weights,
#                     span=self.span,
#                 )
#             else:
#                 # Dummy trend for insufficient cells
#                 def dummy_trend(x):
#                     return np.full_like(x, np.nan)

#                 fit = {"trend": dummy_trend, "std_dev": np.nan}

#             # Extract means and variances
#             xm = x_stats["means"][:, i]
#             xv = x_stats["vars"][:, i]

#             # Calculate technical component
#             tech = np.array([fit["trend"](m) for m in xm])

#             # Calculate biological component
#             bio = xv - tech

#             # Calculate p-values
#             p_value = stats.norm.sf(bio / tech, scale=fit["std_dev"])

#             # Adjust p-values for multiple testing
#             mask = ~np.isnan(p_value)
#             fdr = np.full_like(p_value, np.nan)
#             if np.any(mask):
#                 fdr[mask] = self.adjust_pvalues(p_value[mask])

#             # Create output dictionary
#             output = {
#                 "mean": xm,
#                 "total": xv,
#                 "tech": tech,
#                 "bio": bio,
#                 "p_value": p_value,
#                 "FDR": fdr,
#                 "metadata": {
#                     "mean": fm,
#                     "var": fv,
#                     "trend": fit["trend"],
#                     "std_dev": fit["std_dev"],
#                 },
#             }

#             collected.append(output)

#         return collected

#     def adjust_pvalues(self, p_values: np.ndarray, method: str = "BH") -> np.ndarray:
#         """
#         Adjust p-values for multiple testing.

#         Args:
#             p_values: Raw p-values
#             method: Method for adjustment

#         Returns:
#             Adjusted p-values
#         """
#         p_values = np.asarray(p_values)
#         n = len(p_values)

#         if method == "BH":  # Benjamini-Hochberg procedure
#             # Sort p-values
#             sorted_indices = np.argsort(p_values)
#             sorted_p = p_values[sorted_indices]

#             # Calculate adjusted values
#             adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)

#             # Ensure monotonicity
#             for i in range(n - 2, -1, -1):
#                 adjusted[i] = min(adjusted[i], adjusted[i + 1])

#             # Restore original order
#             result = np.empty_like(adjusted)
#             result[sorted_indices] = adjusted

#             # Cap at 1
#             return np.minimum(result, 1.0)
#         else:
#             raise ValueError(f"Method '{method}' not supported")

#     def _combine_blocks(
#         self,
#         collected: List[Dict[str, Any]],
#         method: str = "fisher",
#         equiweight: bool = True,
#         ncells: Optional[np.ndarray] = None,
#     ) -> Dict[str, Any]:
#         """
#         Combine statistics across blocks.

#         Args:
#             collected: List of dicts with statistics by block
#             method: Method for combining p-values
#             equiweight: Whether to give equal weight to each block
#             ncells: Number of cells in each block

#         Returns:
#             Dict with combined statistics
#         """
#         # Check if any blocks exist
#         if len(collected) == 0:
#             return {}

#         # Get dimensions
#         nblocks = len(collected)
#         ngenes = len(collected[0]["mean"])

#         # Set weights
#         if equiweight:
#             weights = np.ones(nblocks)
#         else:
#             weights = ncells if ncells is not None else np.ones(nblocks)

#         # Normalize weights
#         weights = weights / np.sum(weights)

#         # Initialize output
#         fields = ["mean", "total", "tech", "bio"]
#         output = {}
#         for field in fields:
#             output[field] = np.zeros(ngenes)

#         # Validity mask (for blocks with enough cells)
#         valid = (
#             np.array([n >= 2 for n in ncells])
#             if ncells is not None
#             else np.ones(nblocks, dtype=bool)
#         )

#         # Combine statistics
#         for field in fields:
#             for i in range(nblocks):
#                 if valid[i]:
#                     output[field] += weights[i] * collected[i][field]

#         # Combine p-values
#         if method == "fisher":
#             # Fisher's method
#             p_values = np.zeros(ngenes)
#             for i in range(ngenes):
#                 # Extract p-values for current gene
#                 ps = [collected[j]["p_value"][i] for j in range(nblocks) if valid[j]]

#                 if len(ps) > 0:
#                     # Fisher's method: -2 * sum(log(p))
#                     statistic = -2 * np.sum(np.log(ps))
#                     df = 2 * len(ps)
#                     p_values[i] = 1 - stats.chi2.cdf(statistic, df)
#                 else:
#                     p_values[i] = np.nan
#         else:
#             # Default to minimum p-value (Bonferroni-adjusted)
#             p_values = np.ones(ngenes)
#             for i in range(ngenes):
#                 ps = [collected[j]["p_value"][i] for j in range(nblocks) if valid[j]]
#                 if len(ps) > 0:
#                     p_values[i] = min(ps) * len(ps)  # Bonferroni adjustment
#                 else:
#                     p_values[i] = np.nan

#         # Calculate FDR
#         mask = ~np.isnan(p_values)
#         fdr = np.full_like(p_values, np.nan)
#         if np.any(mask):
#             fdr[mask] = self.adjust_pvalues(p_values[mask])

#         # Add to output
#         output["p_value"] = p_values
#         output["FDR"] = fdr

#         # Add per-block data
#         output["per_block"] = collected
#         output["metadata"] = collected[0]["metadata"] if len(collected) > 0 else {}

#         return output

#     def get_highly_variable(
#         self, n_top: Optional[int] = None, min_bio: float = 0.0
#     ) -> np.ndarray:
#         """
#         Get indices of highly variable genes.

#         Args:
#             n_top: Number of top genes to return
#             min_bio: Minimum biological component

#         Returns:
#             Indices of highly variable genes
#         """
#         if self.results_ is None:
#             raise ValueError("Model not fitted. Call fit() first.")

#         if n_top is not None:
#             # Get top n genes by biological component
#             indices = np.argsort(self.results_["bio"])[::-1][:n_top]
#         else:
#             # Get genes with biological component > threshold
#             indices = np.where(self.results_["bio"] > min_bio)[0]

#         return indices

#     def to_dataframe(self) -> pd.DataFrame:
#         """
#         Convert results to DataFrame.

#         Returns:
#             DataFrame with results
#         """
#         if self.results_ is None:
#             raise ValueError("Model not fitted. Call fit() first.")

#         df = pd.DataFrame(
#             {
#                 "mean": self.results_["mean"],
#                 "total": self.results_["total"],
#                 "bio": self.results_["bio"],
#                 "tech": self.results_["tech"],
#                 "p_value": self.results_["p_value"],
#                 "FDR": self.results_["FDR"],
#             }
#         )

#         if "index" in self.results_:
#             df.index = self.results_["index"]

#         return df

#     def plot(
#         self,
#         ax=None,
#         show_trend: bool = True,
#         highlight_hvg: bool = True,
#         n_top: int = 100,
#         **kwargs,
#     ) -> Any:
#         """
#         Plot the results.

#         Args:
#             ax: Matplotlib axis
#             show_trend: Whether to show the trend line
#             highlight_hvg: Whether to highlight highly variable genes
#             n_top: Number of top genes to highlight
#             **kwargs: Additional arguments for plotting

#         Returns:
#             Matplotlib axis
#         """
#         import matplotlib.pyplot as plt

#         if self.results_ is None:
#             raise ValueError("Model not fitted. Call fit() first.")

#         if ax is None:
#             _, ax = plt.subplots(figsize=(8, 6))

#         # Plot all genes
#         ax.scatter(
#             self.results_["mean"],
#             self.results_["total"],
#             s=3,
#             alpha=0.4,
#             color="gray",
#             **kwargs,
#         )

#         # Highlight HVGs
#         if highlight_hvg:
#             hvg_indices = self.get_highly_variable(n_top=n_top)
#             ax.scatter(
#                 self.results_["mean"][hvg_indices],
#                 self.results_["total"][hvg_indices],
#                 s=10,
#                 alpha=0.6,
#                 color="red",
#                 **kwargs,
#             )

#         # Show trend line
#         if show_trend and "metadata" in self.results_:
#             mean_grid = np.logspace(
#                 np.log10(max(np.min(self.results_["mean"]), 1e-6)),
#                 np.log10(np.max(self.results_["mean"])),
#                 100,
#             )
#             trend_values = [self.results_["metadata"]["trend"](x) for x in mean_grid]
#             ax.plot(mean_grid, trend_values, color="blue", linestyle="-", linewidth=2)

#         ax.set_xlabel("Mean expression")
#         ax.set_ylabel("Variance")
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         ax.grid(True, alpha=0.3)

#         return ax


# class GeneVarianceModel(BaseVarianceModel):
#     """
#     Model for fitting gene variance.

#     This class implements the standard gene variance modeling approach,
#     decomposing gene variance into technical and biological components
#     based on a fitted mean-variance trend.
#     """

#     def __init__(
#         self,
#         min_mean: float = 0.1,
#         trend_method: TrendFitMethod = TrendFitMethod.BOTH,
#         span: float = 0.3,
#         use_density_weights: bool = True,
#         **kwargs,
#     ):
#         """
#         Initialize the model.

#         Args:
#             min_mean: Minimum mean to use for trend fitting
#             trend_method: Method for trend fitting
#             span: Span parameter for LOWESS
#             use_density_weights: Whether to use inverse density weights
#             **kwargs: Additional arguments
#         """
#         super().__init__(min_mean, trend_method, span, use_density_weights)
#         self.kwargs = kwargs

#     def fit(
#         self,
#         X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
#         block: Optional[np.ndarray] = None,
#         design: Optional[np.ndarray] = None,
#         subset_row: Optional[np.ndarray] = None,
#         subset_fit: Optional[np.ndarray] = None,
#         equiweight: bool = True,
#         method: str = "fisher",
#         **kwargs,
#     ) -> "GeneVarianceModel":
#         """
#         Fit the model.

#         Args:
#             X: Expression matrix (genes x cells)
#             block: Blocking factor
#             design: Design matrix
#             subset_row: Rows to use
#             subset_fit: Rows to use for fitting
#             equiweight: Whether to give equal weight to blocks
#             method: Method for combining p-values
#             **kwargs: Additional arguments

#         Returns:
#             Self
#         """
#         # Convert pandas DataFrame to numpy array if needed
#         if isinstance(X, pd.DataFrame):
#             values = X.values
#             index = X.index
#         else:
#             values = X
#             index = None

#         # Get statistics for the data
#         x_stats = self._compute_stats(
#             values, block=block, design=design, subset_row=subset_row
#         )

#         # Get statistics for fitting
#         if subset_fit is None:
#             fit_stats = x_stats
#         else:
#             fit_stats = self._compute_stats(
#                 values, block=block, design=design, subset_row=subset_fit
#             )

#         # Decompose variance
#         collected = self._decompose_variance(x_stats, fit_stats)

#         # Combine statistics
#         self.results_ = self._combine_blocks(
#             collected, method=method, equiweight=equiweight, ncells=x_stats["ncells"]
#         )

#         # Set index
#         if index is not None:
#             if subset_row is not None:
#                 self.results_["index"] = index[subset_row]
#             else:
#                 self.results_["index"] = index

#         return self


# class PoissonVarianceModel(GeneVarianceModel):
#     """
#     Model for fitting gene variance with Poisson noise.

#     This class implements the Poisson-based gene variance modeling approach,
#     using simulated Poisson counts to model the technical variation.
#     """

#     def __init__(
#         self,
#         min_mean: float = 0.1,
#         trend_method: TrendFitMethod = TrendFitMethod.BOTH,
#         span: float = 0.3,
#         use_density_weights: bool = True,
#         npts: int = 1000,
#         dispersion: float = 0,
#         pseudo_count: float = 1.0,
#         **kwargs,
#     ):
#         """
#         Initialize the model.

#         Args:
#             min_mean: Minimum mean to use for trend fitting
#             trend_method: Method for trend fitting
#             span: Span parameter for LOWESS
#             use_density_weights: Whether to use inverse density weights
#             npts: Number of interpolation points for Poisson simulation
#             dispersion: Dispersion parameter (0 for Poisson)
#             pseudo_count: Pseudo-count for log-transformation
#             **kwargs: Additional arguments
#         """
#         super().__init__(min_mean, trend_method, span, use_density_weights, **kwargs)
#         self.npts = npts
#         self.dispersion = dispersion
#         self.pseudo_count = pseudo_count

#     def generate_poisson_values(
#         self,
#         means: np.ndarray,
#         size_factors: np.ndarray,
#         block: Optional[np.ndarray] = None,
#         design: Optional[np.ndarray] = None,
#     ) -> Dict[str, np.ndarray]:
#         """
#         Generate simulated Poisson or negative binomial values.

#         Args:
#             means: Range of mean counts
#             size_factors: Size factors for scaling
#             block: Blocking factor
#             design: Design matrix

#         Returns:
#             Dict with means and variances
#         """
#         # Filter means
#         means = np.asarray(means)
#         means = means[means > 0]

#         # Create interpolation points
#         min_mean = np.min(means)
#         max_mean = np.max(means)
#         pts = np.exp(np.linspace(np.log(min_mean), np.log(max_mean), self.npts))

#         # Create counts matrix with expected values
#         expected_counts = np.outer(pts, size_factors)

#         # Create AnnData object with cells in rows, features in columns
#         adata = AnnData(X=expected_counts)

#         # Generate Poisson or negative binomial counts
#         if self.dispersion == 0:
#             # Poisson distribution
#             adata.X = np.random.poisson(adata.X)
#         else:
#             # Negative binomial distribution
#             size = 1.0 / self.dispersion
#             prob = size / (size + adata.X)
#             adata.X = np.random.negative_binomial(size, prob)

#         # Apply log transformation
#         sc.pp.log1p(adata)

#         # Compute mean and variance statistics (transpose for our format)
#         return self._compute_stats(adata.X.T, block=block, design=design)

#     def fit(
#         self,
#         X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
#         size_factors: Optional[np.ndarray] = None,
#         block: Optional[np.ndarray] = None,
#         design: Optional[np.ndarray] = None,
#         subset_row: Optional[np.ndarray] = None,
#         equiweight: bool = True,
#         method: str = "fisher",
#         **kwargs,
#     ) -> "PoissonVarianceModel":
#         """
#         Fit the model.

#         Args:
#             X: Count matrix (genes x cells)
#             size_factors: Size factors for scaling
#             block: Blocking factor
#             design: Design matrix
#             subset_row: Rows to use
#             equiweight: Whether to give equal weight to blocks
#             method: Method for combining p-values
#             **kwargs: Additional arguments

#         Returns:
#             Self
#         """
#         # Convert pandas DataFrame to numpy array if needed
#         if isinstance(X, pd.DataFrame):
#             values = X.values
#             index = X.index
#         else:
#             values = X
#             index = None

#         # Calculate size factors if not provided
#         if size_factors is None:
#             size_factors = np.sum(values, axis=0)
#             size_factors = size_factors / np.mean(size_factors)

#         # Apply log transformation
#         transformed = log_transform(values, size_factors, self.pseudo_count)

#         # Get statistics for the transformed data
#         x_stats = self._compute_stats(
#             transformed, block=block, design=design, subset_row=subset_row
#         )

#         # Generate range for simulation
#         xlim = (
#             2
#             ** np.array(
#                 [
#                     np.nanmin(x_stats["means"][x_stats["means"] > 0]),
#                     np.nanmax(x_stats["means"]),
#                 ]
#             )
#             - self.pseudo_count
#         )

#         # Simulate Poisson values
#         sim_stats = self.generate_poisson_values(
#             xlim, size_factors, block=block, design=design
#         )

#         # Decompose variance
#         collected = self._decompose_variance(x_stats, sim_stats)

#         # Combine statistics
#         self.results_ = self._combine_blocks(
#             collected, method=method, equiweight=equiweight, ncells=x_stats["ncells"]
#         )

#         # Set index
#         if index is not None:
#             if subset_row is not None:
#                 self.results_["index"] = index[subset_row]
#             else:
#                 self.results_["index"] = index

#         return self


# def model_gene_var(
#     X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
#     block: Optional[np.ndarray] = None,
#     design: Optional[np.ndarray] = None,
#     subset_row: Optional[np.ndarray] = None,
#     subset_fit: Optional[np.ndarray] = None,
#     min_mean: float = 0.1,
#     trend_method: str = "both",
#     span: float = 0.3,
#     use_density_weights: bool = True,
#     equiweight: bool = True,
#     method: str = "fisher",
#     return_model: bool = False,
#     **kwargs,
# ) -> Union[pd.DataFrame, GeneVarianceModel]:
#     """
#     Model the variance of log-expression profiles.

#     Args:
#         X: Expression matrix (genes x cells)
#         block: Blocking factor
#         design: Design matrix
#         subset_row: Rows to use
#         subset_fit: Rows to use for fitting
#         min_mean: Minimum mean to use for trend fitting
#         trend_method: Method for trend fitting ('parametric', 'lowess', or 'both')
#         span: Span parameter for LOWESS
#         use_density_weights: Whether to use inverse density weights
#         equiweight: Whether to give equal weight to blocks
#         method: Method for combining p-values
#         return_model: Whether to return the model object
#         **kwargs: Additional arguments

#     Returns:
#         DataFrame with results or model object
#     """
#     # Convert trend_method string to enum
#     if trend_method.lower() == "parametric":
#         trend_enum = TrendFitMethod.PARAMETRIC
#     elif trend_method.lower() == "lowess":
#         trend_enum = TrendFitMethod.LOWESS
#     else:
#         trend_enum = TrendFitMethod.BOTH

#     # Create and fit model
#     model = GeneVarianceModel(
#         min_mean=min_mean,
#         trend_method=trend_enum,
#         span=span,
#         use_density_weights=use_density_weights,
#         **kwargs,
#     )

#     model.fit(
#         X=X,
#         block=block,
#         design=design,
#         subset_row=subset_row,
#         subset_fit=subset_fit,
#         equiweight=equiweight,
#         method=method,
#     )

#     # Return model or DataFrame
#     if return_model:
#         return model
#     else:
#         return model.to_dataframe()


# def model_gene_var_by_poisson(
#     X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
#     size_factors: Optional[np.ndarray] = None,
#     block: Optional[np.ndarray] = None,
#     design: Optional[np.ndarray] = None,
#     subset_row: Optional[np.ndarray] = None,
#     min_mean: float = 0.1,
#     trend_method: str = "both",
#     span: float = 0.3,
#     use_density_weights: bool = True,
#     equiweight: bool = True,
#     method: str = "fisher",
#     npts: int = 1000,
#     dispersion: float = 0,
#     pseudo_count: float = 1.0,
#     return_model: bool = False,
#     **kwargs,
# ) -> Union[pd.DataFrame, PoissonVarianceModel]:
#     """
#     Model the variance with Poisson noise.

#     Args:
#         X: Count matrix (genes x cells)
#         size_factors: Size factors for scaling
#         block: Blocking factor
#         design: Design matrix
#         subset_row: Rows to use
#         min_mean: Minimum mean to use for trend fitting
#         trend_method: Method for trend fitting ('parametric', 'lowess', or 'both')
#         span: Span parameter for LOWESS
#         use_density_weights: Whether to use inverse density weights
#         equiweight: Whether to give equal weight to blocks
#         method: Method for combining p-values
#         npts: Number of interpolation points for Poisson simulation
#         dispersion: Dispersion parameter (0 for Poisson)
#         pseudo_count: Pseudo-count for log-transformation
#         return_model: Whether to return the model object
#         **kwargs: Additional arguments

#     Returns:
#         DataFrame with results or model object
#     """
#     # Convert trend_method string to enum
#     if trend_method.lower() == "parametric":
#         trend_enum = TrendFitMethod.PARAMETRIC
#     elif trend_method.lower() == "lowess":
#         trend_enum = TrendFitMethod.LOWESS
#     else:
#         trend_enum = TrendFitMethod.BOTH

#     # Create and fit model
#     model = PoissonVarianceModel(
#         min_mean=min_mean,
#         trend_method=trend_enum,
#         span=span,
#         use_density_weights=use_density_weights,
#         npts=npts,
#         dispersion=dispersion,
#         pseudo_count=pseudo_count,
#         **kwargs,
#     )

#     model.fit(
#         X=X,
#         size_factors=size_factors,
#         block=block,
#         design=design,
#         subset_row=subset_row,
#         equiweight=equiweight,
#         method=method,
#     )

#     # Return model or DataFrame
#     if return_model:
#         return model
#     else:
#         return model.to_dataframe()
