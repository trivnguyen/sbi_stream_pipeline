"""Pre-transform pipeline for the NPE stream model.

Adapted from `jgnn.transforms.pipeline`, which owns the composition order
this pipeline has to extend. Only the composition lives here: the
individual transforms are imported from jgnn unchanged, so fixes to them
still propagate. The one addition is `track` -- the unperturbed stream
track, which has to run *first*, before the measurement model and before
graph construction.

Order matters in two ways.

Before the uncertainty sampler: the projection subtracts a deterministic
function of `phi1`, and `phi1` carries no uncertainty, so
`(y + eps) - mu(phi1) == (y - mu(phi1)) + eps`. Either order gives the
same distribution and leaves the recorded sigmas valid; running it first
just makes that independence explicit. If `phi1` ever gains an
uncertainty, the projection has to move after the noise and use the
observed `phi1`.

Before graph construction: with `project_pos`, the kNN graph is built
from the straightened stream rather than the bent one.
"""

import torch
from torch_geometric import transforms as T
from torch_geometric.loader import DataLoader as PyGDataLoader

from jgnn.transforms.basic import GetNodeFeatures, Normalize
from jgnn.transforms.graph import ALL_GRAPHS
from jgnn.transforms.projection import RandomProjection
from jgnn.transforms.selection_function import RandomSelectionStrategy
from jgnn.transforms.uncertainty import UncertaintySampler


def build_transformation(
    apply_graph: bool = True,
    apply_projection: bool = False,
    apply_selection: bool = False,
    apply_uncertainty: bool = False,
    recompute_node_features: bool = False,
    graph_name: str = 'KNN',
    graph_args: dict = None,
    projection_args: dict = None,
    selection_args: dict = None,
    uncertainty_args=None,
    norm_dict=None,
    use_log_features: bool = True,
    track=None,
):
    """Build the pre-transform pipeline for graph data.

    Identical to `jgnn.transforms.build_transformation` except for
    `track`, a `TrackProjection` (or None) applied ahead of everything
    else. See the module docstring for why it goes first.

    Raises:
        ValueError: If `track` is combined with `recompute_node_features`,
            which would rebuild `x` from `pos`/`vel` and discard the
            projection.
    """
    if track is not None and recompute_node_features:
        raise ValueError(
            'recompute_node_features rebuilds x from pos/vel, discarding '
            'the track projection; set it False when using a track')

    transforms = []
    transforms.append(T.ToDevice(device=torch.device("cpu")))  # not gpu-supported yet

    if track is not None:
        transforms.append(track)

    # Apply random projection and/or selection function
    if apply_projection:
        transforms.append(RandomProjection(**projection_args))
    if apply_selection:
        if selection_args is None:
            raise ValueError('`selection_args` must be provided when `apply_selection` is True.')
        transforms.append(RandomSelectionStrategy(**selection_args))
    if recompute_node_features:
        transforms.append(GetNodeFeatures(log=use_log_features))

    # Apply uncertainty sampling
    if apply_uncertainty:
        if uncertainty_args is None:
            raise ValueError('`uncertainty_args` must be provided when `apply_uncertainty` is True.')
        if isinstance(uncertainty_args, dict):
            uncertainty_args = [uncertainty_args]
        for args in uncertainty_args:
            transforms.append(UncertaintySampler(**args))

    # Normalizing node features
    if norm_dict is not None:
        transforms.append(Normalize(norm_dict['x_loc'], norm_dict['x_scale']))

    # Apply graph transformation, connect edges based on the specified graph
    # type; set to False for no graph construction (e.g. Transformer models)
    if apply_graph:
        if graph_name.lower() not in ALL_GRAPHS:
            raise ValueError(f"Unknown graph name: {graph_name}. Supported graphs: {list(ALL_GRAPHS.keys())}")
        transforms.append(ALL_GRAPHS[graph_name.lower()](**graph_args))

    return T.Compose(transforms)


def compute_norm_dict(graphs, batch_size: int = 256, track=None,
                      **pre_transform_kwargs):
    """Compute `x` normalization stats from the real pre-transform pipeline.

    As `jgnn.transforms.compute_norm_dict`, but through the pipeline
    above, so the stats describe the *projected* features when a track is
    in use. Graph construction and `Normalize` are skipped, since only
    `x` is needed.

    Note this measures mean/std, not a min/max range -- so the track's
    reduction in spread feeds straight into the scale, rather than being
    hostage to the most extreme stream in the set.

    Args:
        graphs: List of PyG `Data` objects.
        batch_size: Batch size used while streaming `graphs` through.
        track: `TrackProjection` to apply first, or None.
        **pre_transform_kwargs: Same kwargs as `build_transformation`.

    Returns:
        Tuple of `(x_loc, x_scale)` tensors.
    """
    kwargs = dict(pre_transform_kwargs)
    kwargs['apply_graph'] = False
    kwargs['norm_dict'] = None
    pipeline = build_transformation(track=track, **kwargs)

    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    x_all = torch.cat([pipeline(batch).x for batch in loader], dim=0)

    return x_all.mean(dim=0), x_all.std(dim=0)
