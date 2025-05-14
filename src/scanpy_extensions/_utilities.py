import contextlib
from collections.abc import Iterable
from typing import Any, Mapping, Union

import joblib
import scanpy as sc
from packaging.version import Version

from ._validate import isiterable

# handle random
if Version(sc.__version__) >= Version("1.11.0"):
    from scanpy._utils import SeedLike as RandomState
else:
    from scanpy._utils import AnyRandom as RandomState


def session_info() -> None:
    import datetime
    import sys
    from subprocess import check_output

    print("*" * 64)
    print(f"Execution date and time: {datetime.datetime.now()}")
    print("*" * 64)
    print("CPU model: ", end="")
    print(
        check_output("lscpu | grep 'Model name'", shell=True, text=True)
        .split(":")[1]
        .strip()
    )
    print("*" * 64)
    print(f"Anaconda environment name: {sys.executable.split('/')[-3]}")
    print("*" * 64)


def set_env(print_info: bool = True) -> None:
    sc.settings.verbosity = 4
    sc.settings.n_jobs = 8
    if print_info:
        session_info()


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


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def gene_liftover(
    adata: sc.AnnData,
    gene_list: Iterable[str],
    gene_alias_dict: dict[str, Iterable[str]],
) -> list[str]:
    from scanpy import logging as logg

    val_gene_list = list()
    val_gene_set = set()
    adata_genes = set(adata.var_names.to_list())
    for x in gene_list:
        found = False
        alias = False
        if x in adata_genes:
            if x not in val_gene_set:
                val_gene_set.add(x)
                val_gene_list.append(x)
            found = True
        elif x in gene_alias_dict:
            alias = True
            for y in gene_alias_dict[x]:
                if y in adata_genes:
                    if y not in val_gene_set:
                        val_gene_set.add(y)
                        val_gene_list.append(y)
                    found = True
                    logg.warning(f"Using alias {y} for {x}")
        if not found:
            logg.warning(f"{x} does not exist")
            if alias:
                logg.warning(f"aliases {gene_alias_dict[x]} also does not exist")
    return val_gene_list


def create_scenic_adata(
    adata: sc.AnnData,
    obsm_keys: Union[str, Iterable[str]] = ["X_umap", "X_tsne"],
    format_regulon_names: bool = True,
) -> sc.AnnData:
    reg_idx = adata.obs.columns.str.startswith("Regulon")
    pos_reg_idx = adata.obs.columns.str.match("Regulon.*\(\+\)\)")
    regs = adata.obs.columns[pos_reg_idx]
    _obsm = {x: adata.obsm[x] for x in obsm_keys if x in adata.obsm.keys()}
    rdata = sc.AnnData(
        X=adata.obs[regs],
        obs=adata.obs[adata.obs.columns[~reg_idx]],
        obsm=_obsm,
        dtype=str(adata.obs[regs].dtypes[0]),
    )
    if format_regulon_names:
        rdata.var_names = rdata.var_names.map(lambda x: str(x)[8:-4])
    return rdata


__all__ = [
    "RandomState",
    "session_info",
    "set_env",
    "update_config",
    "tqdm_joblib",
    "gene_liftover",
    "create_scenic_adata",
]
