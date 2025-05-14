from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Union

import numba
import numpy as np


@numba.njit()
def _weighted_median(
    x: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> float:
    if w is None:
        return np.median(x)
    else:
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        w_cum = np.cumsum(w[sorted_idx])
        w_total = w_cum[-1]

        med_idx = np.searchsorted(w_cum, (w_total / 2))
        if med_idx >= (len(x) - 1):
            return x_sorted[-1]
        elif w_cum[med_idx] == (w_total / 2):
            return np.mean(x_sorted[med_idx : med_idx + 1])
        else:
            return x_sorted[med_idx]


def weighted_median(x: np.ndarray, w: np.ndarray, na_rm: bool = False) -> float:
    _x = np.asarray(x)
    _w = np.asarray(w)
    if na_rm:
        mask = ~np.isnan(_x)
        _x = _x[mask]
        _w = _w[mask]

    return _weighted_median(_x, _w)


def inverse_density_weights(
    x: np.ndarray,
    bw_method: Union[Literal["scott", "silverman"], float] = "silverman",
) -> np.ndarray:
    from scipy.stats import gaussian_kde

    _x = np.asarray(x)
    density = gaussian_kde(_x, bw_method=bw_method)(x)
    w = 1.0 / np.clip(density, a_min=1e-10, a_max=None)
    return w / np.mean(w)


def weighted_lowess(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
    span: float = 0.3,
    iter: int = 3,
) -> Dict[str, np.ndarray]:
    from skmisc import loess

    _x = np.asarray(x)
    _y = np.asarray(y)
    _w = None if w is None else np.asarray(w)
    params = dict(weights=_w, span=span, degree=1, iterations=iter)

    # Create and fit the LOESS model
    model = loess.loess(_x, _y, **params)
    model.fit()
    fitted = model.predict(_x, stderror=True).values

    # Return a dictionary with fitted values and other info
    return {
        "fitted": fitted,
        "residual": y - fitted,
        "x": x,
        "y": y,
        "weights": w,
    }


class LOWESSTrendFitter:
    """Class for fitting mean-variance trends."""

    do_parametric: bool = True
    do_lowess: bool = True
    use_density_weights: bool = True
    span: float = 0.3

    def __init__(
        self,
        flavor: Literal["parametric", "lowess", "both"] = "both",
        use_density_weights: bool = True,
        span: float = 0.3,
    ):
        assert flavor in ["parametric", "lowess", "both"], (
            f"invalid 'flavor' provided: {flavor}."
        )
        assert (span > 0.0) and (span <= 1.0), f"'span' must be between 0 and 1: {span}"
        if flavor == "parametric":
            self.do_lowess = False
        elif flavor == "lowess":
            self.do_parametric = False
        self.use_density_weights = use_density_weights
        self.span = span

    @staticmethod
    def correct_logged_expectation(
        x: np.ndarray, y: np.ndarray, w: np.ndarray, func: Callable
    ) -> Dict[str, Any]:
        """
        Adjust for scale shift due to fitting to log-values.

        Args:
            x: x values
            y: y values
            w: Weights
            func: Function to generate trend

        Returns:
            Dict with trend function and standard deviation
        """

        # Calculate leftovers
        func_vals = [func(x) for x in np.asarray(x)]
        leftovers = y / func_vals

        # Calculate weighted median
        med = weighted_median(leftovers, w, na_rm=True)

        return dict(
            trend_func=(lambda x: func(x) * med),
            std_dev=(
                weighted_median(np.abs(leftovers / med - 1), w, na_rm=True) * 1.4826
            ),
        )

    @staticmethod
    def get_init_params(
        vars: np.ndarray,
        means: np.ndarray,
        left_n: int = 100,
        left_prop: float = 0.1,
        grid_length: int = 10,
        b_grid_range: float = 5,
        n_grid_max: float = 7,
    ) -> Dict[str, float]:
        """
        Get starting parameters for non-linear curve fitting.

        Args:
            vars: Variances
            means: Means
            left_n: Number of points to use from left
            left_prop: Proportion of points to use from left
            grid_length: Number of grid points
            b_grid_range: Range for B parameter grid
            n_grid_max: Maximum value for n parameter grid

        Returns:
            Dict with starting parameters
        """

        # Sort by means
        n = len(vars)
        sorted_idx = np.argsort(means)

        # Estimate gradient from left
        _left_n = min(left_n, int(n * left_prop))
        keep_idx = sorted_idx[: max(1, _left_n)]
        _vars = vars[keep_idx]
        _means = means[keep_idx]

        # Linear regression through origin
        slope = np.sum(_means * _vars) / np.sum(_means**2)

        # Grid search for remaining parameters
        b_grid, n_grid = np.meshgrid(
            np.exp2(np.linspace(-b_grid_range, b_grid_range, grid_length)),
            np.exp2(np.linspace(0, n_grid_max, grid_length)),
        )
        b_flat = b_grid.flatten()
        n_flat = n_grid.flatten()

        # Evaluate sum of squares for each parameter combination
        best_ss = np.inf
        best_idx = 0
        for i in range(b_flat.shape[0]):
            _b = b_flat[i]
            _n = n_flat[i]
            pred = (slope * _b * means) / (_b + np.power(means, _n))
            resd = vars - pred
            _ss = np.dot(resd, resd)
            if _ss < best_ss:
                best_ss = _ss
                best_idx = i
        return dict(
            n=max(1e-8, n_flat[best_idx] - 1),
            b=b_flat[best_idx],
            a=b_flat[best_idx] * slope,
        )

    @staticmethod
    def _nls_model(x: float, a: float, b: float, n: float):
        # Define the model function: y = (a*x)/(x^(1+n) + b)
        return (a * x) / (b + (x ** (1 + n)))

    @staticmethod
    def _default_param(x: float, y: float):
        return min(1, (x / y))

    @staticmethod
    def _lowess_unscale(x: float, loess_func: Callable, param_func: Callable):
        return np.exp(loess_func(x)) * param_func(x)

    def fit_trend_var(
        self,
        means: np.ndarray,
        vars: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        assert len(vars) >= 2, "need at least 2 points for fitting."
        w = (
            inverse_density_weights(means, bw_method=1.0)
            if self.use_density_weights
            else None
        )

        # Default parametric trend is a straight line from 0 to min(m)
        to_fit = np.log(vars)
        left_edge = np.min(means)
        _param_func = partial(LOWESSTrendFitter._default_param, y=left_edge)

        # Fit parametric curve if requested
        if self.do_parametric:
            assert len(vars) >= 4, (
                "need at least 4 points for non-linear curve fitting."
            )
            # Get initial parameters
            model_params = LOWESSTrendFitter.get_init_params(vars, means)
            try:
                from scipy import optimize

                opt_params, _ = optimize.curve_fit(
                    LOWESSTrendFitter._nls_model,
                    means,
                    vars,
                    p0=[model_params["a"], model_params["b"], model_params["n"]],
                    sigma=(None if w is None else (1 / np.sqrt(w))),
                    method="trf",
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    max_nfev=500,
                )
                _param_func = partial(
                    LOWESSTrendFitter._nls_model,
                    a=opt_params[0],
                    b=opt_params[1],
                    n=opt_params[2],
                )
            except RuntimeError:
                # Use initial estimates if fitting fails
                _param_func = partial(
                    LOWESSTrendFitter._nls_model,
                    a=model_params["a"],
                    b=model_params["b"],
                    n=(model_params["n"] + 1),
                )
            # Update to_fit with residuals
            fitted_values = np.array([_param_func(x) for x in means])
            to_fit = to_fit - np.log(fitted_values)

        _unscaled_func = _param_func
        if self.do_lowess:
            from scipy.interpolate import PchipInterpolator

            lfit = weighted_lowess(means, to_fit, w=w, span=self.span, **kwargs)
            sorted_idx = np.argsort(means)
            _unscaled_func = partial(
                LOWESSTrendFitter._lowess_unscale,
                loess_func=PchipInterpolator(
                    x=means[sorted_idx], y=lfit["fitted"][sorted_idx], extrapolate=False
                ),
                param_func=_param_func,
            )

        # Adjust for scale shift
        return LOWESSTrendFitter.correct_logged_expectation(
            means, vars, w, _unscaled_func
        )
