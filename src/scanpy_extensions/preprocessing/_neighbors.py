from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import scanpy as sc
from scanpy import logging as logg
from scipy.sparse import coo_matrix, csr_matrix

from .._utilities import update_config
from .._validate import validate_groupby
from ..get import obs_categories


def _sparse_dist_to_ind_and_dist(
    D: csr_matrix, n_neighbors: int
) -> Tuple[np.ndarray[np.int32], np.ndarray[np.float32]]:
    indices = np.zeros((D.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=D.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0
        # account for the fact that there might be more than n_neighbors
        # due to an approximate search
        # [the point itself was not detected as its own neighbor during the search]
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][
                neighbors[0][sorted_indices], neighbors[1][sorted_indices]
            ]
        else:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
    return indices, distances


def _calc_conn_and_dist(
    adata: sc.AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    method: Literal["sklearn", "pynnd"] = "pynnd",
    **kwargs,
) -> Tuple[csr_matrix, csr_matrix]:
    from scanpy.tools._utils import _choose_representation
    from umap.umap_ import fuzzy_simplicial_set

    if n_neighbors > adata.shape[0]:  # very small datasets
        n_neighbors = 1 + int(0.5 * adata.shape[0])
        logg.warning(f"`n_obs` too small: adjusting to `n_neighbors = {n_neighbors}`")

    _n_pcs = min(30, adata.obsm[use_rep].shape[1]) if n_pcs is None else n_pcs
    rep = _choose_representation(adata, use_rep=use_rep, n_pcs=_n_pcs)

    knt_params = dict(**kwargs)
    update_config("n_jobs", sc.settings.n_jobs, knt_params)
    if method == "sklearn":
        from sklearn.neighbors import KNeighborsTransformer

        knt = KNeighborsTransformer(
            n_neighbors=n_neighbors + 1, mode="distance", metric=metric, **knt_params
        )
    elif method == "pynnd":
        from pynndescent import PyNNDescentTransformer

        update_config("random_state", 0, knt_params)
        knt = PyNNDescentTransformer(
            n_neighbors=n_neighbors + 1, metric=metric, **knt_params
        )

    knn_ind, knn_dist = _sparse_dist_to_ind_and_dist(
        knt.fit_transform(rep), n_neighbors=n_neighbors
    )
    conn, _, _, dist = fuzzy_simplicial_set(
        coo_matrix(([], ([], [])), shape=(adata.shape[0], 1)),
        n_neighbors,
        None,
        None,
        knn_indices=knn_ind,
        knn_dists=knn_dist,
        return_dists=True,
    )
    conn = conn if isinstance(conn, csr_matrix) else conn.tocsr()
    dist = dist if isinstance(dist, csr_matrix) else dist.tocsr()

    return (conn, dist)


def _extract_sparse_data(
    D: coo_matrix,
    data: Dict[str, list[float]],
    idx_map: np.ndarray[int],
) -> None:
    data["data"] += list(D.data.ravel())
    data["row"] += list(idx_map[D.row.ravel()])
    data["col"] += list(idx_map[D.col.ravel()])


def get_conn_and_dist(
    adata: sc.AnnData,
    n_neighbors: Optional[int] = None,
    n_pcs: Optional[int] = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    method: Literal["sklearn", "pynnd"] = "pynnd",
    only_conn: bool = False,
    groupby: Optional[str] = None,
    **kwargs,
) -> Union[csr_matrix, Tuple[csr_matrix, csr_matrix]]:
    assert groupby is None or validate_groupby(adata, groupby)

    cell_counts = (
        adata.shape[0] if groupby is None else adata.obs[groupby].value_counts().mean()
    )
    _n_neighbors = (
        15 + int(np.round(np.log2(cell_counts))) if n_neighbors is None else n_neighbors
    )
    if groupby is None:
        conn, dist = _calc_conn_and_dist(
            adata,
            n_neighbors=_n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
            metric=metric,
            method=method,
            **kwargs,
        )
    else:
        from scipy.sparse import csr_matrix

        cats = obs_categories(adata, groupby)
        _conn_data = dict(data=[], row=[], col=[])
        _dist_data = dict(data=[], row=[], col=[])
        for c in cats:
            _subset = adata.obs[groupby] == c
            _subset_to_og_idx = np.argwhere(_subset).ravel()
            _adata = adata[_subset].copy()
            _conn, _dist = _calc_conn_and_dist(
                _adata,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_rep=use_rep,
                metric=metric,
                **kwargs,
            )
            _extract_sparse_data(
                _conn.tocoo(), data=_conn_data, idx_map=_subset_to_og_idx
            )
            _extract_sparse_data(
                _dist.tocoo(), data=_dist_data, idx_map=_subset_to_og_idx
            )
        conn = csr_matrix(
            (_conn_data["data"], (_conn_data["row"], _conn_data["col"])),
            shape=(adata.shape[0], adata.shape[0]),
            dtype=_conn.dtype,
        )
        dist = csr_matrix(
            (_dist_data["data"], (_dist_data["row"], _dist_data["col"])),
            shape=(adata.shape[0], adata.shape[0]),
            dtype=_dist.dtype,
        )

    if only_conn:
        return conn
    else:
        return (conn, dist)


def sklearn_neighbors(
    adata: sc.AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    key_added: Optional[str] = None,
) -> None:
    start = logg.info("computing neighbors")

    n_key = "neighbors" if key_added is None else key_added
    d_key = "distances" if key_added is None else f"{key_added}_distances"
    c_key = "connectivities" if key_added is None else f"{key_added}_connectivities"

    conn, dist = _calc_conn_and_dist(
        adata=adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        method="sklearn",
    )

    logg.debug("computed neighbors", time=start)

    adata.obsp[c_key] = conn
    adata.obsp[d_key] = dist
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
