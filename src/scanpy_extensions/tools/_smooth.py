from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import logging as logg

from .._validate import validate_groupby, validate_keys, validate_layer_and_raw
from ..preprocessing._neighbors import get_conn_and_dist


def _smooth_over_graph(
    val: pd.Series,
    graph: Iterable[float],
    z_score: bool = False,
    scale: bool = False,
    na_fill: float = 0.0,
    undo_log: bool = True,
) -> pd.Series:
    from scipy.sparse import issparse

    _val = val.copy() if isinstance(val, pd.Series) else val.squeeze()
    _val[np.isnan(_val)] = na_fill
    _sparse_g = issparse(graph)
    g = graph.copy()
    g_data = g.data if _sparse_g else g
    g_data[np.isnan(g_data)] = 0.0
    if undo_log:
        _val = np.expm1(_val)
    x = np.array(_val).squeeze()
    x = (np.expand_dims(x, 0) @ g).ravel()
    if z_score:
        x = (x - np.mean(x)) / np.std(x)
    x = (np.expand_dims(x, 0) @ g).ravel()
    if undo_log:
        x = np.log1p(x)
    if scale:
        _og_min = np.min(val)
        _og_range = np.max(val) - _og_min
        _min = np.min(x)
        _range = np.max(x) - _min
        x = (((x - _min) / _range) * _og_range) + _og_min
    _dtype = "float64" if str(_val.dtype)[-2:] == "64" else "float32"
    return pd.Series(x, index=val.index).astype(_dtype)


def smooth_over_neighbors(
    adata: sc.AnnData,
    key: str,
    groupby: Optional[str] = None,
    key_added: Optional[str] = None,
    z_score: bool = False,
    scale: bool = False,
    undo_log: Optional[bool] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    na_fill: float = 0.0,
    obsp_key: Optional[str] = None,
    n_neighbors: Optional[int] = None,
    n_pcs: Optional[int] = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    inplace: bool = True,
    **kwargs,
) -> Optional[pd.Series]:
    from scanpy.get import obs_df

    assert obsp_key is None or obsp_key in adata.obsp.keys(), (
        f"'{obsp_key}' not in .obsp."
    )
    assert groupby is None or validate_groupby(adata, groupby)
    assert sum([z_score, scale]) < 2, "cannot specify both 'z_score' and 'scale'."

    start = logg.info(f"computing smooth over neighbors for {key}.")

    _layer, _use_raw = validate_layer_and_raw(adata, layer, use_raw)
    validate_keys(adata, key)
    _undo_log = (
        (True if key not in adata.obs.columns else False)
        if undo_log is None
        else undo_log
    )

    if obsp_key is None:
        start_neighbors = logg.info("computing neighbors.")

        _n_graph = get_conn_and_dist(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
            metric=metric,
            only_conn=True,
            groupby=groupby,
            **kwargs,
        )

        logg.debug("computed neighbors.", time=start_neighbors)
    else:
        _n_graph = adata.obsp[obsp_key]

    smooth_val = _smooth_over_graph(
        obs_df(adata, key, layer=_layer, use_raw=_use_raw),
        _n_graph,
        z_score=z_score,
        scale=scale,
        na_fill=na_fill,
        undo_log=_undo_log,
    )

    logg.debug(f"computed {key} smoothed over neighbors.", time=start)
    if inplace:
        _key = f"{key}_son" if key_added is None else key_added
        logg.debug(f"added to obs['{_key}'].", time=start)
        adata.obs[_key] = smooth_val
    else:
        return smooth_val
