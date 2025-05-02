from typing import Optional

import numpy as np
import scanpy as sc
from scanpy import logging as logg


def sklearn_neighbors(
    adata: sc.AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    key_added: Optional[str] = None,
) -> None:
    from scanpy.tools._utils import _choose_representation
    from scipy.sparse import coo_matrix, csr_matrix
    from sklearn.neighbors import KNeighborsTransformer
    from umap.umap_ import fuzzy_simplicial_set

    start = logg.info("computing neighbors")

    n_key = "neighbors" if key_added is None else key_added
    d_key = "distances" if key_added is None else f"{key_added}_distances"
    c_key = "connectivities" if key_added is None else f"{key_added}_connectivities"

    if n_neighbors > adata.shape[0]:  # very small datasets
        n_neighbors = 1 + int(0.5 * adata.shape[0])
        logg.warning(f"`n_obs` too small: adjusting to `n_neighbors = {n_neighbors}`")

    rep = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    dist, ind = (
        KNeighborsTransformer(
            n_neighbors=n_neighbors + 1,
            mode="distance",
            metric=metric,
        )
        .fit(rep)
        .kneighbors(rep, n_neighbors=n_neighbors + 1)
    )
    start_connect = logg.debug("computed neighbors", time=start)

    conn, _, _ = fuzzy_simplicial_set(
        coo_matrix(([], ([], [])), shape=(adata.shape[0], 1)),
        n_neighbors,
        None,
        None,
        knn_indices=ind[:, 1:].copy(),
        knn_dists=dist[:, 1:].copy(),
    )
    logg.debug("computed connectivities", time=start_connect)

    adata.obsp[c_key] = conn.tocsr()
    adata.obsp[d_key] = csr_matrix(
        (
            dist[:, 1:].ravel(),
            (np.repeat(np.arange(adata.shape[0]), n_neighbors), ind[:, 1:].ravel()),
        ),
        shape=(adata.shape[0], adata.shape[0]),
    )
    adata.uns[n_key] = {}
    adata.uns[n_key]["connectivities_key"] = c_key
    adata.uns[n_key]["distances_key"] = d_key
    adata.uns[n_key]["params"] = dict(
        n_neighbors=n_neighbors,
        method="umap",
        metric=metric,
        use_rep=use_rep,
    )
    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{n_key!r}]`\n"
            f"    `.obsp[{d_key!r}]`, distances for each pair of neighbors\n"
            f"    `.obsp[{c_key!r}]`, weighted adjacency matrix"
        ),
    )

    return
