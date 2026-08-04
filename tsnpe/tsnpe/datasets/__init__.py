"""Stream dataset loading for the TSNPE rounds.

Local rather than `jgnn.datasets` for the same reason `tsnpe.transforms`
is local: jgnn's Cartesian loader reads the dsph layout and has no
`feat_labels`, so it cannot read the files
`stream_sims.sims.write_graph_dataset` produces. Mirrors `npe/datasets`.
"""

from . import cartesian
from .io import read_graph_dataset, read_datasets

__all__ = [
    'cartesian',
    'read_graph_dataset',
    'read_datasets',
]
