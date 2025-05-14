from collections.abc import Iterable, Mapping
from typing import Optional, Union

import pandas as pd
import scanpy as sc
from scanpy import logging as logg

from ._validate import isiterable, ismapping, validate_layer_and_raw


def obs_categories(
    adata: sc.AnnData,
    key: str,
) -> Iterable[str]:
    from pandas.api.types import is_categorical_dtype

    return list(
        adata.obs[key].cat.categories
        if is_categorical_dtype(adata.obs[key])
        else adata.obs[key].unique()
    )


def obs_data(
    adata: sc.AnnData,
    keys: Union[str, Iterable[str]],
    layer: Optional[str] = None,
    use_raw: Optional[str] = None,
    as_series: Optional[bool] = None,
) -> Union[pd.Series, pd.DataFrame]:
    from scanpy.get import obs_df

    feats = keys if isiterable(keys) else [keys]
    _as_series = (
        as_series if isinstance(as_series, bool) else True if len(feats) == 1 else False
    )
    _layer, _use_raw = validate_layer_and_raw(adata, layer=layer, use_raw=use_raw)
    ret = obs_df(adata, feats, layer=_layer, use_raw=_use_raw)
    if _as_series:
        if len(feats) == 1:
            return ret.squeeze()
        else:
            logg.warning("Using DataFrame as multiple keys have been requested.")
    return ret.to_frame() if isinstance(ret, pd.Series) else ret


def sample_fractions(
    adata: sc.AnnData,
    keys: str,
    groupby: Union[str, tuple[str, str]],
    norm: bool = True,
    totals: Union[Iterable[float], Mapping[str, float]] = None,
    dropna: bool = False,
    return_grouped_fractions: bool = False,
) -> pd.DataFrame:
    groups = groupby if isiterable(groupby) else [groupby]
    single_groupby = len(groups) == 1
    main_g = groups[0] if single_groupby else groups[1]
    g_cats = obs_categories(adata, main_g)
    f_cats = obs_categories(adata, keys)
    assert totals is None or len(g_cats) == len(totals), (
        f"Length of provided totals ({len(totals)}) does not match length of categories ({len(g_cats)})."
    )
    df = pd.crosstab(
        index=adata.obs[main_g],
        columns=adata.obs[keys],
        dropna=dropna,
        normalize=(0 if (norm and totals is None) else False),
    )
    df = df.loc[g_cats, f_cats]
    if norm and totals is not None:
        _totals = [totals[x] for x in g_cats] if ismapping(totals) else totals
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
            .groupby([groups[0], keys], observed=False, dropna=dropna)
            .mean()
            .reset_index(),
            index=groups[0],
            columns=keys,
        )
        df.columns = [x[1] for x in df.columns]
        df = df.loc[obs_categories(adata, groups[0]), f_cats]
    df.index.name = groups[0]
    df.columns.name = keys
    return df
