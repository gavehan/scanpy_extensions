from collections.abc import Iterable
from typing import Literal, Optional

import numpy as np
import scanpy as sc
from numba import jit
from scanpy import logging as logg
from scanpy._utils import AnyRandom

from .._utilities import update_config
from .._validate import validate_keys, validate_layer_and_raw


def __get_indices(ref, vals):
    if not isinstance(ref, np.ndarray):
        ref = np.array(ref)
    sorter = np.argsort(ref)
    return sorter[np.searchsorted(ref, vals, sorter=sorter)]


# # Code adapted from
# # https://github.com/aertslab/ctxcore/blob/main/src/ctxcore/recovery.py
# def __uw_auc1d(ranks, rank_threshold, dtype):
#     _ranks = np.sort(ranks[ranks < rank_threshold])
#     _ranks = np.concatenate((_ranks, np.array((rank_threshold,), dtype=np.int32)))
#     _weights = np.arange(_ranks.size - 1, dtype=dtype) + np.cast[dtype](1.0)
#     return np.sum(np.diff(_ranks) * _weights)


# Code adapted from
# https://github.com/aertslab/ctxcore/blob/main/src/ctxcore/recovery.py
@jit(nopython=True)
def __w_auc1d(ranks, rank_threshold, weights):
    _idx = ranks < rank_threshold
    _ranks = ranks[_idx]
    _weights = weights[_idx]
    _idx = np.argsort(_ranks)
    _ranks = np.concatenate(
        (_ranks[_idx], np.full((1,), rank_threshold, dtype=np.int32))
    )
    _weights = _weights[_idx].cumsum()
    return np.sum(np.diff(_ranks) * _weights)


# Code adapted from
# https://github.com/YosefLab/visionpy/blob/main/src/visionpy/signature.py
def __vision(X, sig_matrix, sig_df):
    from scanpy.preprocessing._utils import _get_mean_var

    # normalize
    mean, var = _get_mean_var(X, axis=1)
    n = np.asarray((sig_matrix > 0.0).sum(0))
    m = np.asarray((sig_matrix < 0.0).sum(0))
    sig_df = sig_df / (n + m)
    # cells by signatures
    sig_mean = np.outer(mean, ((n - m) / (n + m)))
    # cells by signatures
    sig_std = np.sqrt(np.outer(var, 1 / (n + m)))
    sig_df = (sig_df - sig_mean) / sig_std
    return np.ravel(sig_df["vision"])


# Code adapted from
# https://github.com/scverse/scanpy/blob/master/scanpy/tools/_score_genes.py
def gene_signature(
    adata: sc.AnnData,
    gene_list: Iterable[str],
    weights: Optional[Iterable[float]] = None,
    flavor: Literal["vision", "aucell"] = "aucell",
    score_name: str = "score",
    random_state: AnyRandom = 0,
    copy: bool = False,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    **kwargs,
) -> Optional[sc.AnnData]:
    """\
    Score a set of genes [Satija15]_.
    The score is the average expression of a set of genes subtracted with the
    average expression of a reference set of genes. The reference set is
    randomly sampled from the `gene_pool` for each binned expression value.
    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.
    Parameters
    ----------
    adata
        The annotated data matrix.
    gene_list
        The list of gene names used for score calculation.
    score_name
        Name of the field to be added in `.obs`.
    random_state
        The random seed for sampling.
    copy
        Copy `adata` or modify it inplace.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.

        .. versionchanged:: 1.4.5
           Default value changed from `False` to `None`.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with an additional field
    `score_name`.
    Examples
    --------
    See this `notebook <https://github.com/theislab/scanpy_usage/tree/master/180209_cell_cycle>`.
    https://doi.org/10.1038/s41467-019-12235-0
    https://doi.org/10.1038/nmeth.4463
    """
    import pandas as pd
    import scipy.sparse as sparse
    from scanpy.get import _get_obs_rep

    _layer, _use_raw = validate_layer_and_raw(adata, layer, use_raw)
    assert flavor in ("vision", "aucell"), f"Unrecognized flavor: {flavor}"
    assert len(gene_list) == len(set(gene_list)), (
        f"Gene list has {len(gene_list) - len(set(gene_list))} duplicate entries."
    )
    assert len(gene_list) > 0, "Gene list is empty."
    validate_keys(adata, gene_list)
    assert weights is None or len(gene_list) == len(weights), (
        "Length of weights and gene list is not equal."
    )
    if flavor == "aucell" and weights is not None:
        assert len(weights) == np.sum(weights > 0), (
            "Non-positive weights for flavor 'aucell'."
        )

    _params = dict(kwargs)
    update_config("aucell_threshold", 5e-2, _params)
    assert 0.0 < _params["aucell_threshold"] <= 1.0, (
        "'aucell_threshold' is an invalid value outside of range 0 to 1."
    )
    update_config("aucell_zscore", False, _params)

    start = logg.info(f"computing score {score_name!r}")
    adata = adata.copy() if copy else adata
    if _use_raw:
        X = adata.raw.X
        computed_on = "adata.raw.X"
    elif _layer is not None:
        X = _get_obs_rep(adata, layer=_layer)
        computed_on = _layer
    else:
        X = adata.X
        computed_on = "adata.X"
    _dtype = X.dtype if X.dtype.kind == "f" else "float32"

    var_names = adata.raw.var_names if _use_raw else adata.var_names
    _gene_list = np.array(gene_list)
    _weights = None if weights is None else np.cast[_dtype](np.array(weights))

    if flavor == "vision":
        sig_matrix = np.ones(len(adata.var_names), dtype=_dtype)
        sig_matrix[__get_indices(var_names, _gene_list)] = _weights
        sig_matrix = sparse.csr_matrix(np.expand_dims(sig_matrix, 1))
        cell_signature_matrix = (X @ sig_matrix).toarray()
        sig_df = pd.DataFrame(
            data=cell_signature_matrix, columns=["vision"], index=adata.obs_names
        )
        scores = __vision(X, sig_matrix, sig_df)
    elif flavor == "aucell":
        X_rank = (
            pd.DataFrame(
                np.array(X.toarray()) if sparse.issparse(X) else X,
                columns=adata.var_names,
                index=adata.obs_names,
            )
            .sample(
                frac=1.0,
                replace=False,
                axis=1,
                random_state=random_state,
            )
            .rank(axis=1, ascending=False, method="first", na_option="bottom")
            .astype(np.int32)
            - 1
        )
        X_rank = X_rank.loc[:, X_rank.columns.isin(_gene_list)].copy()
        rank_threshold = np.int32(
            int(round(_params["aucell_threshold"] * X.shape[1])) - 1
        )
        auc_weights = (
            np.ones_like(_gene_list, dtype=_dtype) if _weights is None else _weights
        )
        scores = np.apply_along_axis(
            __w_auc1d,
            axis=1,
            arr=X_rank.to_numpy(),
            rank_threshold=rank_threshold,
            weights=auc_weights,
        )
        auc_max = np.cast[_dtype](np.sum(auc_weights) * (rank_threshold + 1.0))
        scores /= auc_max
        if _params["aucell_zscore"]:
            scores = (scores - np.mean(scores)) / np.std(scores)

    adata.obs[score_name] = pd.Series(np.ravel(scores), index=adata.obs_names)
    adata.uns[score_name] = {
        "flavor": flavor,
        "computed_on": computed_on,
        "params": {"gene_list": list(_gene_list), "random_state": random_state},
    }
    if _weights is not None:
        adata.uns[score_name]["params"]["weights"] = list(_weights)
    if flavor == "aucell":
        adata.uns[score_name]["params"]["auc_threshold"] = _params["aucell_threshold"]
        adata.uns[score_name]["params"]["rank_threshold"] = rank_threshold
        adata.uns[score_name]["params"]["auc_max"] = auc_max
        adata.uns[score_name]["params"]["zscore"] = _params["aucell_zscore"]
    logg.info(
        "    finished",
        time=start,
        deep=(f"added\n    {score_name!r}, score of gene set (adata.obs).\n"),
    )

    return adata if copy else None
