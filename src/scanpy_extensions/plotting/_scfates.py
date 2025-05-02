def __get_binned_minmax(adata, bins, paths, feat, undo_log=False, layer=None):
    min_data = []
    max_data = []
    for i, t in enumerate(paths.keys()):
        _adata = adata[adata.obs["seg"].isin(paths[t])].copy()
        cur_cell_bins = bins[bins.index.isin(_adata.obs.index)]
        x = obs_df(_adata, feat, layer=layer)
        x = x.transform(np.expm1) if undo_log else x
        x = x.groupby(cur_cell_bins, observed=True).mean()
        x = x.transform(np.log1p) if undo_log else x
        min_data.append(x.min(axis=0))
        max_data.append(x.max(axis=0))
    return pd.DataFrame(min_data).min(axis=0), pd.DataFrame(max_data).max(axis=0)


def scf_matrix(
    adata,
    num_keys_map: Mapping[str, Iterable[str]] = None,
    cat_keys: Iterable[str] = None,
    layer: str = "fitted",
    nbins: int = 10,
    pseudotime_key: str = "t",
    add_pseudotime: bool = True,
    no_overlapping_paths: bool = False,
    cat_to_num_ratio: float = 2.0,
    undo_log_keys: Iterable[str] = None,
    text_wrap: int = 25,
    linespacing: float = 0.9,
    pseudotime_colormap: Union[str, mpl.colors.Colormap] = None,
    color_map_map: Mapping[str, Union[str, mpl.colors.Colormap]] = None,
    palette_map: Mapping[str, Union[str, mpl.colors.Colormap]] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
) -> Optional[mpl.figure.Figure]:
    import networkx as nx
    from matplotlib.patches import Patch

    # Handle categorical features
    if cat_keys is not None:
        for _c in cat_keys:
            assert _c in adata.obs.columns and adata.obs[_c].dtype == "category", (
                f"{_c} neither in adata.obs.columns or is not categorical dtype!"
            )
        cat_feats = cat_keys
    else:
        cat_feats = []

    if palette_map is not None:
        assert (
            len(palette_map) <= len(cat_feats)
            and len(set(palette_map.keys()) - set(cat_feats)) == 0
        ), "'palette_map' keys must be a subset to items in 'cat_keys'"
    else:
        palette_map = {}
    palette_map = {
        x: __get_colors(
            adata,
            x,
            palette_map[x] if x in palette_map.keys() else None,
            return_map=True,
        )
        for x in cat_feats
    }

    # Handle numerical features
    num_feats_map = []
    if add_pseudotime:
        num_feats_map.append([pseudotime_key, [pseudotime_key]])
    for k, v in num_keys_map.items():
        num_feats_map.append([k, v])
        assert pseudotime_key not in v, (
            f"Pseudotime key '{pseudotime_key}' should not be provided by 'num_keys_map'. Please use 'add_pseudotime' parameter instead!"
        )
        for _v in v:
            assert _v in adata.obs.columns or _v in adata.var.index, (
                f"{_v} neither in adata.obs.columns or adata.var.index!"
            )
    num_feats_map = OrderedDict(num_feats_map)

    if color_map_map is not None:
        assert (
            len(color_map_map) <= len(num_feats_map)
            and len(set(color_map_map.keys()) - set(num_keys_map.keys())) == 0
        ), "'color_map_map' keys must be a subset to keys of 'num_keys_map'"
    else:
        color_map_map = {}
    color_map_map = {
        x: color_map_map[x] if x in color_map_map.keys() else None
        for x in num_feats_map.keys()
    }
    if add_pseudotime:
        color_map_map[pseudotime_key] = pseudotime_colormap

    # Process graph
    g_info = adata.uns["graph"]
    G = nx.from_pandas_edgelist(
        g_info["pp_seg"][["from", "to", "d"]].rename({"d": "weight"}, axis=1),
        source="from",
        target="to",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )
    lengths, all_paths = nx.single_source_dijkstra(G, g_info["root"])
    lengths = pd.Series(lengths).sort_values()
    g_tips = set(g_info["tips"])
    paths = []
    if no_overlapping_paths:
        for n in lengths.index:
            if n in g_tips:
                continue
            edges = pd.Series(
                {j: a["weight"] for _, j, a in G.edges(n, data=True)}
            ).sort_values()
            is_first = True
            for m in edges.index:
                if m in g_tips:
                    if is_first:
                        is_first = False
                        paths.append((m, [list(G.predecessors(n))[0], n, m]))
                    else:
                        paths.append((m, [n, m]))
    else:
        for n, d in lengths.items():
            if n in g_tips and d > 0.0:
                paths.append((n, all_paths[n]))
    m_to_s_map = {
        (row["from"], row["to"]): row["n"] for _, row in g_info["pp_seg"].iterrows()
    }
    paths = OrderedDict(
        [
            (t, [m_to_s_map[(p[i], p[i + 1])] for i in range(len(p) - 1)])
            for t, p in paths
        ]
    )

    # Bin cells based on pseudotime
    cell_bins = pd.cut(adata.obs[pseudotime_key], bins=nbins)
    cell_bins = cell_bins.loc[adata.obs.index]

    # Min-max normalization for numerical features
    minmax_map = {}
    for num_groups, cur_feats in num_feats_map.items():
        minmax_map[num_groups] = __get_binned_minmax(
            adata,
            cell_bins,
            paths,
            cur_feats,
            undo_log=True if num_groups in undo_log_keys else False,
            layer=layer,
        )
    if add_pseudotime:
        pt_min, pt_max = __get_binned_minmax(adata, cell_bins, paths, [pseudotime_key])

    # Solve layout of figure
    total_feats = [cat_to_num_ratio for _ in range(len(cat_feats))]
    toatl_feats_w_leg = []
    for i in range(len(cat_keys)):
        toatl_feats_w_leg += [0.1, 2]
    for x in num_feats_map.values():
        total_feats.append(len(x))
        toatl_feats_w_leg.append(len(x))

    total_bins = []
    for t in paths.keys():
        _adata = adata[adata.obs["seg"].isin(paths[t])]
        total_bins.append(
            len(cell_bins[cell_bins.index.isin(_adata.obs.index)].unique())
        )

    # Create figure
    fs = mpl.rcParams["figure.figsize"]
    fig = plt.figure(
        layout="constrained",
        figsize=figsize
        if figsize is not None
        else (0.2 * sum(total_feats) * fs[0], 0.1 * sum(total_bins) * fs[1]),
    )
    fig.get_layout_engine().set(**layout_params)
    sm_layout = []
    for i, feat in enumerate(cat_keys):
        sm_layout.append([f"cat_leg_{i}" for _ in range(len(total_bins))])
        sm_layout.append([f"cat_{i}_{j}" for j in range(len(total_bins))])
    for i, num_groups in enumerate(num_feats_map.keys()):
        sm_layout.append([f"num_{i}_{j}" for j in range(len(total_bins))])

    axd = fig.subplot_mosaic(
        sm_layout,
        width_ratios=total_bins,
        height_ratios=toatl_feats_w_leg,
    )

    # Process per 'path'
    for i, t in enumerate(paths.keys()):
        _adata = adata[adata.obs["seg"].isin(paths[t])].copy()
        cur_cell_bins = cell_bins[cell_bins.index.isin(_adata.obs.index)]
        col_share_x_ax = None

        for j, feat in enumerate(cat_keys):
            cur_ax = axd[f"cat_{j}_{i}"]
            if col_share_x_ax is None:
                col_share_x_ax = cur_ax
            else:
                cur_ax.sharex(col_share_x_ax)
            cur_cats = _adata.obs[feat].cat.categories
            pd.crosstab(cur_cell_bins, _adata.obs[feat], normalize=0).plot(
                kind="bar",
                stacked=True,
                ax=cur_ax,
                legend=False,
                color=[palette_map[feat][c] for c in cur_cats],
            )
            cur_ax.set_xlabel("")
            cur_ax.tick_params(
                axis="x", labelbottom=False, labeltop=False, bottom=False, top=False
            )
            if i > 0:
                cur_ax.tick_params(
                    axis="y", labelleft=False, labelright=False, left=False, right=False
                )
            else:
                # cur_ax.set_yticks(
                #     [0.5],
                #     labels=["\n".join(textwrap.wrap(feat, width=text_wrap))],
                #     linespacing=linespacing
                # )
                all_cats = adata.obs[feat].cat.categories.to_list()
                leg_ax = axd[f"cat_leg_{j}"]
                leg_ax.axis("off")
                leg_ax.legend(
                    handles=[Patch(fc="w", ec="none", alpha=0, label=feat)]
                    + [
                        Patch(fc=palette_map[feat][c], ec="none", label=c)
                        for c in all_cats
                    ],
                    ncols=len(all_cats) + 1,
                    mode="expand",
                    borderaxespad=0,
                )

        for j, (num_groups, cur_feats) in enumerate(num_feats_map.items()):
            undo_log = num_groups in undo_log_keys
            cur_ax = axd[f"num_{j}_{i}"]
            if col_share_x_ax is None:
                col_share_x_ax = cur_ax
            else:
                cur_ax.sharex(col_share_x_ax)
            drop_pt = False if num_groups == pseudotime_key else True
            df = obs_df(
                _adata,
                cur_feats + [pseudotime_key] if drop_pt else cur_feats,
                layer=layer,
            ).sort_values(pseudotime_key, ascending=False)
            if drop_pt:
                df.drop(pseudotime_key, axis=1, inplace=True)
            if drop_pt:
                cur_mins, cur_maxs = minmax_map[num_groups]
            else:
                cur_mins, cur_maxs = (pt_min, pt_max)
            df = df.transform(np.expm1) if undo_log else df
            df = df.groupby(cur_cell_bins, observed=True).mean()
            df = df.transform(np.log1p) if undo_log else df
            df = (df - cur_mins) / (cur_maxs - cur_mins)
            cur_ax.imshow(
                df.transpose().to_numpy(),
                vmin=0.0,
                vmax=1.0,
                interpolation="none",
                origin="upper",
                aspect="auto",
                cmap=color_map_map[num_groups],
            )
            cur_ax.tick_params(
                axis="x", labelbottom=False, labeltop=False, bottom=False, top=False
            )
            if i > 0:
                cur_ax.tick_params(
                    axis="y", labelleft=False, labelright=False, left=False, right=False
                )
            else:
                cur_ax.set_yticks(
                    [i for i in range(len(cur_feats))],
                    labels=[
                        "\n".join(textwrap.wrap(x, width=text_wrap)) for x in cur_feats
                    ],
                    linespacing=linespacing,
                )
            cur_ax.grid(False)

    savefig_or_show("scf_matrix", show=show, save=save)
    if show is False:
        return fig


def scf_trajectory(
    adata: sc.AnnData,
    key: str = "t",
    layer: str = "fitted",
    basis: str = None,
    arrows: bool = True,
    arrow_dens: float = 10,
    arrow_dir_offset: int = 3,
    arrow_pos_offset: float = 0.15,
    add_cell_emb: bool = False,
    cell_emb_key: str = None,
    ax: mpl.axes.Axes = None,
    color_map: Union[str, mpl.colors.Colormap] = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    nodes: bool = True,
    node_size: float = 1.0,
    arrow_size: float = 1.0,
    arrow_outline_color: str = None,
    arrow_outline_width: float = 1.0,
    show_spines: bool = True,
    edge_color_kwargs: Mapping[str, Any] = MappingProxyType({}),
    node_kwargs: Mapping[str, Any] = MappingProxyType({}),
    node_label_kwargs: Mapping[str, Any] = MappingProxyType({}),
    cell_emb_kwargs: Mapping[str, Any] = MappingProxyType({}),
    cell_emb_label_kwargs: Mapping[str, Any] = MappingProxyType({}),
    adjusttext_kwargs: Mapping[str, Any] = MappingProxyType({}),
    # textalloc_kwds: Mapping[str, Any] = MappingProxyType({}),
    figure_kwds: Mapping[str, Any] = MappingProxyType({}),
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    return_data: Optional[bool] = None,
) -> Optional[mpl.figure.Figure]:
    import networkx as nx
    import scipy.sparse as sp
    import textalloc as ta
    from matplotlib.collections import LineCollection
    from scipy.spatial.distance import cdist

    _basis = None
    e_msg = None

    if basis is None:
        basis_selector = ("umap", "tsne", "pca")
        for b in basis_selector:
            if b in adata.obsm.keys() or f"X_{b}" in adata.obsm.keys():
                _basis = b
                break
        else:
            e_msg = "No valid embedding basis are present is adata.obsm"
    elif basis in adata.obsm.keys() or f"X_{basis}" in adata.obsm.keys():
        _basis = basis if basis[:2] == "X_" else f"X_{basis}"
    else:
        e_msg = f"{basis} is not a valid embedding basis present is adata.obsm"
    if _basis is None:
        raise KeyError(e_msg)

    R = adata.obsm["X_R"]
    g_info = adata.uns["graph"]
    pp_info = g_info["pp_info"]
    emb = adata.obsm[_basis]
    proj = (np.dot(emb.T, R) / R.sum(axis=0)).T
    pos = dict(zip(pp_info.index, proj))
    m_nodes = np.concatenate([g_info["tips"], g_info["forks"]])
    node_info = {
        x: {int(y): z for y, z in pp_info[x].to_dict().items()}
        for x in pp_info.columns[1:]
    }

    G = nx.from_scipy_sparse_array(sp.csr_array(g_info["B"]))
    if arrows:
        G = nx.bfs_tree(G, g_info["root"])
        lengths = nx.single_source_dijkstra_path_length(G, g_info["root"])
        m_nodes_labels = pd.Series({x: lengths[x] for x in m_nodes}).sort_values()
        m_nodes_labels = dict(zip(m_nodes_labels.index, range(len(m_nodes_labels))))
    else:
        m_nodes_labels = np.array([n for n in G.nodes() if n in m_nodes])
        m_nodes_labels = dict(
            zip([int(x) for x in m_nodes_labels], [str(x) for x in m_nodes_labels])
        )
    for x in node_info.keys():
        nx.set_node_attributes(G, node_info[x], x)

    if arrows:
        d_df = pd.DataFrame(
            cdist(g_info["F"].transpose(), g_info["F"].transpose(), metric="euclidean")
        )
        d_df = (
            d_df.where(np.triu(np.ones(d_df.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        d_pairs = dict(zip(zip(d_df["level_0"], d_df["level_1"]), d_df[0]))
        nx.set_edge_attributes(G, d_pairs, "dist")

    if key is not None:
        # from sklearn.neighbors import RadiusNeighborsTransformer
        feat = np.dot(obs_df(adata, key, layer=layer).to_numpy(), R)
        # x_ = adata.obsm["X_umap"][:, 0].ravel()
        # y_ = adata.obsm["X_umap"][:, 1].ravel()
        # emb_n = RadiusNeighborsTransformer(
        #     mode="connectivity",
        #     radius=np.linalg.norm([(np.max(x_)-np.min(x_)), (np.max(y_)-np.min(y_))])/64,
        # ).fit_transform(proj).toarray()
        # p_emb = (np.linalg.matrix_power(g_info["B"], int(np.round(R.shape[1]/64))) > 0).astype(int)
        # feat = np.dot(np.dot(obs_df(adata, key, layer=layer).to_numpy(), R), (emb_n*p_emb))

        node_feat_map = dict(zip([int(x) for x in pp_info.index], feat))
        for e in G.edges():
            G.edges[e][key] = np.mean([node_feat_map[e[0]], node_feat_map[e[1]]])
        ec_arr = np.array([x[-1] for x in G.edges(data=key)])
        vmin = np.min(ec_arr) if vmin is None else vmin
        vmax = np.max(ec_arr) if vmax is None else vmax
        if vcenter is None:
            cnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            cnorm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots(**figure_kwds, layout="constrained")
        fig.get_layout_engine().set(**layout_params)
    else:
        fig = ax.get_figure()
    if show_spines:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ms = 1.0 + mpl.rcParams["axes.linewidth"] * 2
        ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=ms)
        ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=ms)
    else:
        ax.axis("off")
    ax.grid("off")

    edge_color_kwargs = {**edge_color_kwargs}
    node_kwargs = {**node_kwargs}
    node_label_kwargs = {**node_label_kwargs}
    if "path_effects" not in edge_color_kwargs.keys():
        edge_color_kwargs["path_effects"] = []
    edge_color_kwargs["path_effects"] += [
        mpl.patheffects.Stroke(capstyle="round" if key is None else "butt")
        # mpl.patheffects.Stroke(capstyle="round")
    ]
    node_kwargs["s"] = node_size * 75
    if "c" not in node_kwargs.keys():
        node_kwargs["c"] = "k"
    if "color" not in node_label_kwargs.keys() and "c" not in node_label_kwargs.keys():
        node_label_kwargs["color"] = "w"
    if (
        "horizontalalignment" not in node_label_kwargs.keys()
        and "ha" not in node_label_kwargs.keys()
    ):
        node_label_kwargs["horizontalalignment"] = "center"
    if (
        "verticalalignment" not in node_label_kwargs.keys()
        and "va" not in node_label_kwargs.keys()
    ):
        node_label_kwargs["verticalalignment"] = "center"

    arrow_pairs = []
    for _, row in g_info["pp_seg"].iterrows():
        seg_nodes = nx.shortest_path(G, row["from"], row["to"])
        G_ = G.subgraph(seg_nodes)
        if key is not None:
            lc_params = dict(
                array=[node_feat_map[x] for x in seg_nodes],
                cmap=color_map,
                norm=cnorm,
                **edge_color_kwargs,
            )
        else:
            lc_params = dict(
                array=[1 for _ in range(len(seg_nodes))],
                cmap=color_map,
                **edge_color_kwargs,
            )

        points = proj[seg_nodes, :].copy().reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        ax.add_collection(
            LineCollection(
                segs,
                **lc_params,
                zorder=1.0,
            )
        )
        if arrows:
            narrows = max(
                1, int(np.floor(len(seg_nodes) / pp_info.shape[0] * arrow_dens))
            )
            arr = np.cumsum(
                [x[-1] for x in G_.edges(seg_nodes, data="dist", default=0.0)]
            )
            arr = arr / arr.max()
            for i in range(narrows):
                _val = (i + 1) / (narrows + 1)
                if narrows == 1:
                    _val += arrow_pos_offset
                _val = np.min([_val, 0.99])
                arrow_start = seg_nodes[np.abs(arr - _val).argmin()]
                arrow_pairs.append([arrow_start, list(G_.edges(arrow_start))[0][1]])
    if nodes:
        for n in m_nodes:
            _x, _y = pos[n]
            ax.scatter(
                _x,
                _y,
                marker="o",
                zorder=1.2,
                **node_kwargs,
            )
            ax.text(_x, _y, m_nodes_labels[n], **node_label_kwargs)

    if arrows:
        for ap in arrow_pairs:
            offset_node = ap[1]
            if arrow_dir_offset > 1:
                for _ in range(arrow_dir_offset):
                    offset_node = list(G.edges(offset_node))[0][1]
            quiver_kwargs = dict(
                width=arrow_size * 5,
                headwidth=12,
                headaxislength=16,
                headlength=16,
                units="dots",
                rasterized=True,
                pivot="mid",
                angles="xy",
            )
            if arrow_outline_color is not None:
                quiver_kwargs["path_effects"] = [
                    mpl.patheffects.withStroke(
                        linewidth=arrow_outline_width, foreground=arrow_outline_color
                    ),
                    mpl.patheffects.Stroke(capstyle="round"),
                ]
            ax.quiver(
                pos[ap[0]][0],
                pos[ap[0]][1],
                pos[offset_node][0] - pos[ap[0]][0],
                pos[offset_node][1] - pos[ap[0]][1],
                **quiver_kwargs,
                zorder=1.1,
            )

    add_jitter_label = None
    if add_cell_emb:
        cell_emb_kwargs = {**cell_emb_kwargs}
        cell_emb_label_kwargs = {**cell_emb_label_kwargs}
        textjitter_args = dict(**adjusttext_kwargs)
        if cell_emb_key is not None:
            cell_emb_kwargs["color"] = cell_emb_key
        else:
            cell_emb_kwargs["color"] = None
        cell_emb_kwargs["show"] = False
        if "legend_loc" not in cell_emb_kwargs:
            cell_emb_kwargs["legend_loc"] = "right margin"
        add_jitter_label = (
            (cell_emb_key is not None)
            and (adata.obs[cell_emb_key].dtype == "category")
            and (cell_emb_kwargs["legend_loc"] == "on data")
        )
        if add_jitter_label:
            cell_emb_kwargs["legend_loc"] = None
        x_ = adata.obsm[_basis][:, 0].ravel()
        y_ = adata.obsm[_basis][:, 1].ravel()
        if "avoid_label_lines_overlap" not in textjitter_args:
            textjitter_args["avoid_label_lines_overlap"] = True
        if "seed" not in textjitter_args:
            textjitter_args["seed"] = 2023
        if "linewidth" not in textjitter_args:
            textjitter_args["linewidth"] = mpl.rcParams["axes.linewidth"]
        if "linecolor" not in textjitter_args:
            textjitter_args["linecolor"] = "k"
        # if 'force_text' not in textjitter_args:
        #     textjitter_args['force_text'] = (4e-2, 4e-2)
        # if 'force_static' not in textjitter_args:
        #     textjitter_args['force_static'] = (4e-2, 4e-2)
        # if 'force_explode' not in textjitter_args:
        #     textjitter_args['force_explode'] = (2e-2, 2e-2)
        # if 'force_pull' not in textjitter_args:
        #     textjitter_args['force_pull'] = (1e-3, 1e-3)
        # if 'avoid_self' not in textjitter_args:
        #     textjitter_args['avoid_self'] = True
        # if 'explode_radius' not in textjitter_args:
        #     textjitter_args['explode_radius'] = np.linalg.norm([(np.max(x_)-np.min(x_)), (np.max(y_)-np.min(y_))])/10
        # if 'time_lim' not in textjitter_args:
        #     textjitter_args['time_lim'] = 2.0
        embedding(
            adata,
            basis=_basis,
            ax=ax,
            zorder=-0.2,
            **cell_emb_kwargs,
        )
        if add_jitter_label:
            all_pos = (
                pd.DataFrame(adata.obsm[_basis], columns=["x", "y"])
                .groupby(adata.obs[cell_emb_key].to_numpy(), observed=True)
                .median()
                .sort_index()
            )
            # texts = []
            # for t, row in all_pos.iterrows():
            #     texts.append(
            #         ax.text(row["x"], row["y"], t, **cell_emb_label_kwargs)
            #     )
            if adata.shape[0] > 1e3:
                _idx = gs(adata.obsm[_basis], int(1e3), replace=False, seed=0)
            else:
                _idx = np.arange(adata.shape[0])
            _x = list(x_[_idx])
            _y = list(y_[_idx])
            if nodes:
                for n in m_nodes:
                    _x.append(pos[n][0])
                    _y.append(pos[n][1])
            # adjust_text(
            #     texts,
            #     x=_x,
            #     y=_y,
            #     ax=ax,
            #     **textjitter_args
            # )
            ta.allocate(
                ax=ax,
                x=all_pos["x"].to_numpy(),
                y=all_pos["y"].to_numpy(),
                text_list=all_pos.index.to_numpy(),
                x_scatter=_x,
                y_scatter=_y,
                # **cell_emb_label_kwargs,
                **textjitter_args,
            )

    # Set limits for specific scenario
    if add_jitter_label is None:
        x_min, y_min = emb.min(axis=0)
        x_max, y_max = emb.max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - (2.5e-2 * x_range), x_max + (2.5e-2 * x_range))
        ax.set_ylim(y_min - (2.5e-2 * y_range), y_max + (2.5e-2 * y_range))
    if show_spines:
        xlab, ylab = ax.get_xlabel(), ax.get_ylabel()
        ax.set_xlabel(xlab[2:] if xlab[:2] == "X_" else xlab)
        ax.set_ylabel(ylab[2:] if ylab[:2] == "X_" else ylab)

    savefig_or_show("scf_trajectory", show=show, save=save)
    if show is False:
        if return_data is True:
            return fig, ec_arr
        else:
            return fig
