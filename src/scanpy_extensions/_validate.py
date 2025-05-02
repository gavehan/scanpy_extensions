from collections.abc import Iterable
from typing import Mapping, Optional, Union

import pandas as pd
import scanpy as sc
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_numeric_dtype
from scanpy import logging as logg


def isiterable(x) -> bool:
    return not isinstance(x, str) and isinstance(x, Iterable)


def ismapping(x) -> bool:
    return (
        isinstance(x, Mapping)
        or isinstance(x, pd.Series)
        or isinstance(x, pd.DataFrame)
    )


def validate_keys(
    adata: sc.AnnData, keys: Union[str, Iterable[str]], check_numeric: bool = True
) -> Iterable[bool]:
    _is_num = []
    _keys = keys if isiterable(keys) else [keys]
    for k in _keys:
        _cur_is_num = True
        if k in adata.obs.keys():
            if not (is_numeric_dtype(adata.obs[k]) or is_bool_dtype(adata.obs[k])):
                _cur_is_num = False
                if check_numeric:
                    raise TypeError(
                        f"Key {k} in .obs.columns is not numeric or boolean dtype."
                    )
        elif (k not in adata.var_names) and (
            adata.raw is not None and k not in adata.raw.var_names
        ):
            raise KeyError(f"Could not find key {k} in .var_names or .obs.columns.")
        _is_num.append(_cur_is_num)
    return _is_num


def validate_groupby(adata: sc.AnnData, groupby: Union[str, Iterable[str]]) -> None:
    _groupby = groupby if isiterable(groupby) else [groupby]
    for g in _groupby:
        if g not in adata.obs.keys():
            raise KeyError(f"Could not find key {g} in .obs.columns.")
        elif is_numeric_dtype(adata.obs[g]):
            raise TypeError(f"Key {g} in .obs.columns is numeric dtype.")
        elif not is_categorical_dtype(adata.obs[g]):
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
    return layer, _use_raw
