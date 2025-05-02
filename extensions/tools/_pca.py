from collections.abc import Iterable
from typing import Optional

import matplotlib as mpl
import numpy as np
import scanpy as sc
from scanpy import logging as logg


def evaluate_pca(
    adata: sc.AnnData,
    plot: bool = False,
    uns_key: str = "pca",
    var_thresholds: Iterable[float] = [2, 1.8, 1.5, 1.2],
) -> Optional[mpl.axes.Axes]:
    if uns_key not in adata.uns.keys():
        raise KeyError(f"'{uns_key}' not in .uns.")

    opt_pc = np.sqrt(adata.uns[uns_key]["variance"])
    q = var_thresholds
    print("PC threshold based on standard deviation > ", end="")
    for i in range(len(q)):
        end = "\n" if i == (len(q) - 1) else " | "
        print(f"{q[i]:.1f}: {np.sum(opt_pc > q[i])}", end=end)

    opt_pc = np.cumsum(adata.uns[uns_key]["variance_ratio"])
    max_cum_var = np.max(opt_pc) * 100 // 5 * 5 / 100
    q = [max_cum_var - i * 0.05 for i in range(5)]
    q = [x for x in q if x > 0.0]
    print("PC threshold based on cumulative variance ratio > ", end="")
    for i in range(len(q)):
        end = "\n" if i == (len(q) - 1) else " | "
        print(f"{q[i]:.2f}: {np.sum(opt_pc < q[i])}", end=end)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        _, ax = plt.subplots()
        sns.scatterplot(
            x=list(range(1, len(opt_pc) + 1)),
            y=opt_pc,
            color="k",
            ec="w",
            s=2 + np.log2(plt.rcParams["font.size"]),
            ax=ax,
        )
        for i in range(len(q)):
            ax.axvline(np.sum(opt_pc < q[i]), c="r", ls="--")
        return ax

    return


def pca_from_other(
    s_adata: sc.AnnData,
    o_adata: sc.AnnData,
    n_comps: int,
    use_highly_variable: bool = True,
    zero_center: bool = True,
    key_added: Optional[str] = None,
    copy_compute_layer: bool = False,
    **kwargs,
) -> None:
    from scanpy.get import _get_obs_rep, _set_obs_rep
    from scanpy.preprocessing import pca
    from scipy.sparse import csr_matrix

    assert s_adata.obs_names.equals(o_adata.obs_names), (
        "Source and other anndata obs_names does not match."
    )
    assert len(set(o_adata.var_names) - set(s_adata.var_names)) == 0, (
        "Source and other anndata var_names does not match."
    )
    if use_highly_variable:
        if (
            "highly_variable" in o_adata.var_keys()
            and "highly_variable" in s_adata.var_keys()
        ):
            assert o_adata.var_names[o_adata.var["highly_variable"]].equals(
                s_adata.var_names[s_adata.var["highly_variable"]]
            )
            temp = o_adata[:, o_adata.var["highly_variable"]].copy()
        else:
            raise ValueError(
                "Did not find adata.var['highly_variable']. "
                "Either your data already only consists of highly-variable genes "
                "or consider running `pp.highly_variable_genes` first."
            )
    else:
        temp = o_adata

    logg_start = logg.info(f"computing transferred PCA on with {n_comps}")

    X = _get_obs_rep(temp)
    pca_res = pca(
        X, n_comps=n_comps, zero_center=zero_center, return_info=True, **kwargs
    )

    key_obsm, key_varm, key_uns = (
        ("X_pca", "PCs", "pca") if key_added is None else [key_added] * 3
    )
    s_adata.obsm[key_obsm] = pca_res[0]
    s_adata.uns[key_uns] = {}
    s_adata.uns[key_uns]["params"] = {
        "zero_center": zero_center,
        "use_highly_variable": use_highly_variable,
    }
    if np.sum(pca_res[2]) > 1:
        s_adata.uns[key_uns]["variance_ratio"] = pca_res[3]
        s_adata.uns[key_uns]["variance"] = pca_res[2]
    else:
        s_adata.uns[key_uns]["variance_ratio"] = pca_res[2]
        s_adata.uns[key_uns]["variance"] = pca_res[3]
    if use_highly_variable:
        s_adata.varm[key_varm] = np.zeros(shape=(s_adata.n_vars, n_comps))
        s_adata.varm[key_varm][s_adata.var["highly_variable"]] = pca_res[1].T
    else:
        s_adata.varm[key_varm] = pca_res[1].T

    if copy_compute_layer:
        logg.info("transferring PCA loadings", time=logg_start)
        var_subset = np.isin(
            s_adata.var_names.to_numpy(),
            o_adata.var_names.to_numpy(),
            assume_unique=True,
        )
        s_to_o = np.nonzero(var_subset)[0]
        row = np.empty(X.size, dtype=np.int32)
        col = np.empty(X.size, dtype=np.int32)
        data = np.empty(X.size, dtype=np.float32)
        i = 0
        with np.nditer(X, flags=["multi_index"]) as o_X:
            for val in o_X:
                row[i] = o_X.multi_index[0]
                col[i] = s_to_o[o_X.multi_index[1]]
                data[i] = val
                i += 1
        s_X = csr_matrix((data, (row, col)), shape=s_adata.shape, dtype=np.float32)
        _set_obs_rep(s_adata, s_X)

    logg.info("    finished", time=logg_start)
    logg.debug(
        "and added\n"
        f"    {key_obsm!r}, the PCA coordinates (adata.obs)\n"
        f"    {key_varm!r}, the loadings (adata.varm)\n"
        f"    'pca_variance', the variance / eigenvalues (adata.uns[{key_uns!r}])\n"
        f"    'pca_variance_ratio', the variance ratio (adata.uns[{key_uns!r}])"
    )
    return
