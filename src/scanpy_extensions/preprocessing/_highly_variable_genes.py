"""
Gene variance modeling with simplified scanpy integration.

This module provides classes and functions for modeling gene variance
in single-cell RNA sequencing data, decomposing it into technical and
biological components, using scanpy as the foundation.
"""

from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from scanpy import logging as logg
from scanpy.get import _get_obs_rep
from tqdm import tqdm

from .._utilities import tqdm_joblib
from .._validate import validate_layer_and_raw
from ..get import obs_categories
from ._loess_fit import LOWESSTrendFitter

INFO_COLS = ["means", "variances", "variances_norm"]


def _scran_model_gene_var(
    X,
    min_mean: float = 0.1,
    flavor: Literal["parametric", "lowess", "both"] = "both",
    use_density_weights: bool = True,
    span: float = 0.3,
    index: Optional[Iterable[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    from scanpy.preprocessing._utils import _get_mean_var

    means, vars = _get_mean_var(X)

    # Filter out low-abundance genes
    valid_idx = ~np.isnan(vars) & (vars > 1e-8) & (means >= min_mean)

    _means = means[valid_idx]
    _vars = vars[valid_idx]
    tfit = LOWESSTrendFitter(
        flavor=flavor, use_density_weights=use_density_weights, span=span
    )
    fit_res = tfit.fit_trend_var(means=_means, vars=_vars, **kwargs)

    _var_norm = np.zeros_like(vars)
    _var_norm[valid_idx] = _vars - np.array([fit_res["trend_func"](x) for x in _means])
    return pd.DataFrame(
        dict(
            means=np.asarray(means),
            variances=np.asarray(vars),
            variances_norm=np.asarray(_var_norm),
        ),
        index=index,
    )


def _batched_scran_model_gene_var(
    X,
    current_batch: str,
    index: Optional[Iterable[str]] = None,
    **kwargs,
) -> tuple[str, pd.DataFrame]:
    return (
        current_batch,
        _scran_model_gene_var(X, index=index, **kwargs),
    )


def _get_expr_weighted_latent_coords(
    gene: str, adata: sc.AnnData, obsm_key: str = "X_pca"
):
    ret = pd.Series(sc.get.obs_df(adata, gene) @ adata.obsm[obsm_key])
    ret.name = gene
    return ret


def highly_variable_genes(
    adata: sc.AnnData,
    layer: Optional[str] = None,
    use_raw: Optional[bool] = None,
    n_top_genes: int = 2000,
    min_mean: float = 0.1,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
    batch_key: Optional[str] = None,
    check_coverage: bool = True,
    n_candidate_genes: int = 4000,
    n_pcs: int = 50,
    init_n_levels: int = 20,
    **kwargs,
) -> Optional[pd.DataFrame]:
    if not isinstance(adata, sc.AnnData):
        msg = (
            "`pp.highly_variable_genes` expects an `AnnData` argument, "
            "pass `inplace=False` if you want to return a `pd.DataFrame`."
        )
        raise ValueError(msg)

    _layer, _use_raw = validate_layer_and_raw(adata, layer, use_raw)

    if batch_key is None:
        start = logg.info("extracting highly variable genes")
        X = _get_obs_rep(adata, layer=_layer, use_raw=_use_raw)
        df = _scran_model_gene_var(
            X, min_mean=min_mean, span=span, index=adata.var_names, **kwargs
        )
    else:
        start = logg.info(
            f"extracting highly variable genes per batch using batch key: {batch_key}"
        )
        batches = obs_categories(adata, batch_key)
        with tqdm_joblib(tqdm(total=len(batches), mininterval=0.5, miniters=1)) as _:
            res_collector = Parallel(n_jobs=sc.settings.n_jobs, return_as="generator")(
                delayed(_batched_scran_model_gene_var)(
                    _get_obs_rep(
                        adata[adata.obs[batch_key] == b, :],
                        layer=_layer,
                        use_raw=_use_raw,
                    ),
                    current_batch=b,
                    index=adata.var_names,
                    min_mean=min_mean,
                    span=span,
                    **kwargs,
                )
                for b in batches
            )
            res_collector = dict(res_collector)
        res = dict()
        for x in INFO_COLS:
            res_pool = dict()
            for b in batches:
                res_pool[b] = res_collector[b][x]
            _res_sub = pd.DataFrame(res_pool).mean(axis=1)
            res[x] = _res_sub
        df = pd.DataFrame(res)

    if check_coverage:
        from fastcluster import linkage
        from scipy.cluster.hierarchy import dendrogram

        logg.info(
            "selecting genes based on data coverage from PC space calculated on candidate genes"
        )

        cand_genes = (
            df.loc[(df["variances_norm"] > 0.0)]
            .sort_values("variances_norm", ascending=False)
            .head(n_candidate_genes)
            .index
        )

        temp = sc.AnnData(
            _get_obs_rep(
                adata[:, cand_genes],
                layer=_layer,
                use_raw=_use_raw,
            )
        )
        temp.obs_names = adata.obs_names
        temp.var_names = cand_genes
        sc.pp.scale(temp, max_value=10.0)
        sc.tl.pca(temp, n_comps=n_pcs, use_highly_variable=False)

        logg.info("calculating expression weighted feature coordinates")
        with tqdm_joblib(tqdm(total=temp.shape[1], mininterval=1, miniters=5)) as _:
            gene_coords = Parallel(n_jobs=sc.settings.n_jobs)(
                delayed(_get_expr_weighted_latent_coords)(g, adata=temp)
                for g in temp.var_names
            )
        gene_coords = pd.DataFrame(gene_coords)
        gene_coords_arr = gene_coords.to_numpy(copy=True)

        Z = linkage(gene_coords_arr, method="complete", preserve_input=False)
        p = init_n_levels
        n_coverage_genes = 0
        while n_coverage_genes < n_top_genes:
            d = dendrogram(Z=Z, truncate_mode="level", p=p, no_plot=True)
            gene_idx = [int(x) for x in d["ivl"] if "(" not in x]
            n_coverage_genes = len(gene_idx)

        df["coverage"] = df.index.isin(temp.var_names[gene_idx])
        logg.info(f"total of {len(gene_idx)} genes selected based on data coverage")
        sel_genes = (
            df.loc[df["coverage"], "variances_norm"]
            .sort_values(ascending=False)
            .head(n_top_genes)
            .index
        )
    else:
        sel_genes = (
            df["variances_norm"].sort_values(ascending=False).head(n_top_genes).index
        )
    df["highly_variable"] = df.index.isin(sel_genes)

    logg.info("    finished", time=start)

    if not inplace:
        if subset:
            df = df.loc[df["highly_variable"]]
        return df

    adata.uns["hvg"] = {"flavor": "scran"}
    logg.hint(
        "added\n"
        "    'highly_variable', boolean vector (adata.var)\n"
        "    'means', float vector (adata.var)\n"
        "    'variances', float vector (adata.var)\n"
        "    'variances_norm', float vector (adata.var)"
    )
    adata.var["highly_variable"] = df["highly_variable"]
    adata.var["means"] = df["means"]
    adata.var["variances"] = df["variances"]
    adata.var["variances_norm"] = df["variances_norm"].astype(np.float32, copy=False)

    if check_coverage:
        adata.var["coverage"] = df["coverage"]
    if subset:
        adata._inplace_subset_var(df["highly_variable"])
