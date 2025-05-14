import sys

from . import get
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from ._utilities import (
    create_scenic_adata,
    gene_liftover,
    session_info,
    set_env,
    tqdm_joblib,
)

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pl", "get"]})

__all__ = [
    "create_scenic_adata",
    "gene_liftover",
    "session_info",
    "set_env",
    "tqdm_joblib",
    "pl",
    "tl",
    "get",
]
