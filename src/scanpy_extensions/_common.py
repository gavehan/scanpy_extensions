from collections.abc import Iterable, Mapping
from typing import Any

import scanpy as sc
from packaging.version import Version

# handle random
if Version(sc.__version__) >= Version("1.11.0"):
    from scanpy._utils import AnyRandom as RandomState
else:
    from scanpy._utils import SeedLike as RandomState


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


__all__ = [
    "RandomState",
    "isiterable",
    "update_config",
]
