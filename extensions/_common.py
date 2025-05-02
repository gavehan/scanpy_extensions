from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import pandas as pd
import scanpy as sc
from pandas.api.types import is_numeric_dtype
from scanpy import logging as logg


def isiterable(x) -> bool:
    return not isinstance(x, str) and isinstance(x, Iterable)


def update_config(k: str, v: Any, config: Mapping[str, Any]) -> None:
    k_defined = False
    keys = k if isiterable(k) else [k]
    for _k in keys:
        if _k in config.keys():
            k_defined = True
            if isinstance(v, Mapping):
                for v_k in v.keys():
                    if v_k not in config[_k]:
                        config[_k][v_k] = v[v_k]
            break
    if not k_defined:
        config[keys[0]] = v
    return


def validate_keys(adata: sc.AnnData, keys: Union[str, Iterable[str]]) -> None:
    _keys = keys if isiterable(keys) else [keys]
    for k in _keys:
        if k in adata.obs.keys():
            if not (
                is_numeric_dtype(adata.obs[k]) or str(adata.obs[k].dtype) == "boolean"
            ):
                raise TypeError(
                    f"Key {k} in .obs.columns is not numeric or boolean dtype."
                )
        elif (k not in adata.var_names) and (
            adata.raw is not None and k not in adata.raw.var_names
        ):
            raise KeyError(f"Could not find key {k} in .var_names or .obs.columns.")
    return


def validate_groupby(adata: sc.AnnData, groupby: Union[str, Iterable[str]]) -> None:
    _groupby = groupby if isiterable(groupby) else [groupby]
    for g in _groupby:
        if g not in adata.obs.keys():
            raise KeyError(f"Could not find key {g} in .obs.columns.")
        elif is_numeric_dtype(adata.obs[g]):
            raise TypeError(f"Key {g} in .obs.columns is numeric dtype.")
        elif str(adata.obs[g].dtype) != "category":
            logg.warning(f"Key {g} in .obs.columns is not 'category' dtype.")
    return


def validate_layer_and_raw(
    adata: sc.AnnData,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
) -> tuple[Optional[str], bool]:
    _use_raw = (
        use_raw
        if use_raw is isinstance(use_raw, bool)
        else (True if (layer is None and adata.raw is not None) else False)
    )
    assert not _use_raw or layer is None, (
        "Cannot specify use_raw=True and a layer at the same time."
    )
    if use_raw:
        logg.info("Using .raw.")
    return layer, use_raw


def get_categories(
    adata: sc.AnnData,
    key: str,
) -> Iterable[str]:
    return list(
        adata.obs[key].cat.categories
        if str(adata.obs[key].dtype) == "category"
        else adata.obs[key].unique()
    )


def get_obs_data(
    adata: sc.AnnData,
    keys: Union[str, Iterable[str]],
    layer: Optional[str] = None,
    use_raw: Optional[str] = None,
    as_series: Optional[bool] = None,
) -> Union[pd.Series, pd.DataFrame]:
    from scanpy.get import obs_df

    feats = keys if isiterable(keys) else [keys]
    _as_series = (
        as_series
        if as_series is isinstance(as_series, bool)
        else True
        if len(feats) == 1
        else False
    )
    _layer, _use_raw = validate_layer_and_raw(adata, layer=layer, use_raw=use_raw)
    ret = obs_df(adata, feats, layer=_layer, use_raw=_use_raw)
    if _as_series:
        if len(feats) == 1:
            return ret.squeeze()
        else:
            logg.warning("Using DataFrame as multiple keys have been requested.")
    return ret


def get_fractions(
    adata: sc.AnnData,
    keys: str,
    groupby: Union[str, tuple[str, str]],
    norm: bool = True,
    totals: Union[Iterable[float], Mapping[str, float]] = None,
    dropna: bool = False,
    return_grouped_fractions: bool = False,
):
    groups = groupby if isiterable(groupby) else [groupby]
    single_groupby = len(groups) == 1
    main_g = groups[0] if single_groupby else groups[1]
    g_cats = get_categories(adata, main_g)
    f_cats = get_categories(adata, keys)
    if totals is not None and len(g_cats) != len(totals):
        print("error")
    df = pd.crosstab(
        index=adata.obs[main_g],
        columns=adata.obs[keys],
        dropna=dropna,
        normalize=(0 if (norm and totals is None) else None),
    )
    df = df.loc[g_cats, f_cats]
    if norm and totals is not None:
        _totals = [totals[x] for x in g_cats] if isinstance(totals, Mapping) else totals
        df = (df.transpose() / _totals).transpose()
    if not single_groupby:
        df = pd.melt(df.reset_index(), id_vars=groups[1])
        g2g_map = dict(zip(adata.obs[groups[1]], adata.obs[groups[0]]))
        df[groups[0]] = df[groups[1]].map(g2g_map).astype("category")
        df = df.rename(dict(value="frac"), axis=1)
        if return_grouped_fractions:
            return df
        df = pd.pivot(
            df[[groups[0], keys, "frac"]]
            .groupby([groups[0], keys])
            .mean()
            .reset_index(),
            index=groups[0],
            columns=keys,
        )
        df.columns = [x[1] for x in df.columns]
        df = df.loc[get_categories(adata, groups[0]), f_cats]
    df.index.name = groups[0]
    df.columns.name = keys
    return df
