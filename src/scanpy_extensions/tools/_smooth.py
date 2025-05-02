from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from .._validate import validate_keys, validate_layer_and_raw


def _smooth_over_graph(
    val: pd.Series,
    graph: Iterable[float],
    z_score: bool = False,
    scale: bool = False,
    na_fill: float = 0.0,
    undo_log: bool = True,
) -> pd.Series:
    from scipy.sparse import issparse

    _val = val.copy()
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
    obsp_key: Optional[str] = None,
    key_added: Optional[str] = None,
    z_score: bool = False,
    scale: bool = False,
    undo_log: Optional[bool] = None,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    na_fill: float = 0.0,
    inplace: bool = True,
) -> Optional[pd.Series]:
    from scanpy.get import obs_df

    assert sum([z_score, scale]) < 2, "cannot specify both 'z_score' and 'scale'."

    _layer, _use_raw = validate_layer_and_raw(adata, layer, use_raw)
    validate_keys(adata, key)
    _undo_log = (
        (True if key not in adata.obs.columns else False)
        if undo_log is None
        else undo_log
    )
    _obsp_key = "connectivities" if obsp_key is None else obsp_key

    _n_graph = adata.obsp[_obsp_key]
    val = obs_df(adata, key, layer=_layer, use_raw=_use_raw)
    ret = _smooth_over_graph(
        val, _n_graph, z_score=z_score, scale=scale, na_fill=na_fill, undo_log=_undo_log
    )

    if inplace:
        _key = f"{key}_son" if key_added is None else key_added
        adata.obs[_key] = ret
    else:
        return ret
