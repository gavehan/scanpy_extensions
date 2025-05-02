from collections.abc import Iterable, Mapping
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import logging as logg

from .._validate import isiterable, validate_groupby
from ..get import get_categories, get_fractions


def __cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / (np.std(np.concatenate([x, y]), ddof=-2))


def __hedges_g(x, y):
    _div = (((len(x) - 1) * np.std(x, ddof=2)) + ((len(y) - 1) * np.std(y, ddof=2))) / (
        len(x) + len(y) - 2
    )
    return (np.mean(x) - np.mean(y)) / _div


def effect_sizes(
    adata: sc.AnnData,
    keys: str,
    groupby: tuple[str, str],
    totals: Optional[Union[Iterable[float], Mapping[str, float]]] = None,
    group_order: Optional[Iterable[str]] = None,
    method: Literal["d", "g"] = "g",
) -> pd.DataFrame:
    gkey = keys
    validate_groupby(adata, list(groupby) + [gkey])
    gkey_cats = get_categories(adata, gkey)
    assert isiterable(groupby) and len(groupby) == 2, (
        "'groupby' is not a tuple of length 2."
    )
    main_g = groupby[0]
    main_cats = (
        group_order if group_order is not None else get_categories(adata, main_g)
    )
    assert len(main_cats) == 2, f"'{main_g}' does not have exactly two categories."

    logg_start = logg.info(
        f"computing effect sizes for {groupby[1]} grouped by {main_g} using method {method}"
    )
    logg.info(f"comparing {main_cats[1]} over {main_cats[0]}")

    df = get_fractions(
        adata, gkey, groupby, totals=totals, return_grouped_fractions=True
    )
    res = {}
    for c in gkey_cats:
        _x = df.loc[((df[gkey] == c) & (df[main_g] == main_cats[1])), "frac"].to_numpy()
        _y = df.loc[((df[gkey] == c) & (df[main_g] == main_cats[0])), "frac"].to_numpy()
        res[c] = __hedges_g(_x, _y) if method == "g" else __cohen_d(_x, _y)

    res = pd.Series(res).to_frame().loc[gkey_cats]
    res.columns = ["effect_size"]
    logg.info("    finished", time=logg_start)

    return res
