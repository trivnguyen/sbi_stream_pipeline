"""Cartesian dataset helpers.

Handles HDF5 files produced by sample_galaxies.py whose node features are
3-D Cartesian phase-space coordinates (pos, vel, vel_error).

Graph layout
------------
  pos       : (N, 3)  3-D position, kpc
  vel       : (N, 3)  3-D velocity, km/s
  vel_error : (N, 3)  velocity uncertainty, km/s (optional)
  x         : built by the pre_transform pipeline (GetNodeFeatures and any
              projection/selection/uncertainty transforms), not stored here
  theta     : (1, L)  graph-level labels (parameters)
  cond      : (1, C)  graph-level conditionals (optional)
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from jgnn.transforms import compute_norm_dict


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_graph_from_posvel(feat, pos, label=None):
    """Build a bare PyG graph holding raw Cartesian phase-space data."""
    pos = torch.tensor(pos, dtype=torch.float32)
    feat = torch.tensor(feat, dtype=torch.float32)
    if label is not None:
        label = torch.tensor(label, dtype=torch.float32)

    if pos.dim() == 1:
        pos = pos.view(-1, 1)
    if feat.dim() == 1:
        feat = feat.view(-1, 1)
    if label is not None and label.dim() == 1:
        label = label.view(1, -1)

    return Data(x=feat, pos=pos, theta=label)


# ---------------------------------------------------------------------------
# Internal helper: build graph list from node/graph feature dicts
# ---------------------------------------------------------------------------

def _build_graphs(node_feats, graph_feats, feat_labels, labels, max_graphs=None):
    num_graphs = len(graph_feats['num_particles'])
    if max_graphs is not None:
        num_graphs = min(num_graphs, max_graphs)

    ptr = np.cumsum(graph_feats['num_particles'])
    ptr = np.insert(ptr, 0, 0)

    graphs = []
    loop = tqdm(range(num_graphs), miniters=max(1, num_graphs // 100),
                desc='Building graphs')
    for i in loop:
        sl = slice(ptr[i], ptr[i + 1])
        pos = np.array([node_feats['phi1'][sl], node_feats['phi2'][sl]]).T
        feat = np.array([node_feats[k][sl] for k in feat_labels]).T
        flow_labels = [graph_feats[k][i] for k in labels]

        graph = create_graph_from_posvel(feat=feat, pos=pos, label=flow_labels)
        graphs.append(graph)

    return graphs


def _compute_norm(graphs, pre_transform_kwargs):
    """Compute normalization stats from a list of graphs.

    `x_loc`/`x_scale` come from actually running `graphs` through the
    pre_transform pipeline built from `pre_transform_kwargs` (see
    `jgnn.transforms.compute_norm_dict`), rather than approximating them
    from raw `pos`/`vel`.
    """
    x_loc, x_scale = compute_norm_dict(graphs, **pre_transform_kwargs)

    theta_all = torch.cat([g.theta for g in graphs], dim=0)
    theta_min = theta_all.min(dim=0)[0]
    theta_max = theta_all.max(dim=0)[0]
    theta_loc = (theta_max + theta_min) / 2
    theta_scale = (theta_max - theta_min) / 2

    norm = {
        'x_loc': x_loc.tolist(),
        'x_scale': x_scale.tolist(),
        'theta_loc': theta_loc.tolist(),
        'theta_scale': theta_scale.tolist(),
    }

    return norm


def _apply_norm(graphs, norm_dict, device, cond_labels=None):
    """Normalise theta (and cond) in-place from a norm_dict.

    `x` is normalised later by `Normalize` inside the pre_transform
    pipeline, so it isn't touched here.
    """
    theta_loc = torch.tensor(
        norm_dict['theta_loc'], dtype=torch.float32, device=device)
    theta_scale = torch.tensor(
        norm_dict['theta_scale'], dtype=torch.float32, device=device)
    for g in graphs:
        g.theta = (g.theta - theta_loc) / theta_scale


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    node_feats, graph_feats, feat_labels, labels, train_frac=0.8, train_batch_size=32,
    eval_batch_size=32, num_workers=1, norm_dict=None, seed=0, pre_transform_kwargs=None
):
    """Prepare train/val dataloaders from Cartesian phase-space node features.

    `pre_transform_kwargs` are the same kwargs passed to
    `jgnn.transforms.build_transformation` (minus `norm_dict`). They're
    required whenever `norm_dict` isn't already provided, since computing
    `x_loc`/`x_scale` means running the training graphs through that exact
    pipeline.
    """
    pl.seed_everything(seed)

    graphs = _build_graphs(node_feats, graph_feats, feat_labels, labels)

    num_train = int(len(graphs) * train_frac)
    np.random.shuffle(graphs)
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]

    device = train_graphs[0].pos.device

    if norm_dict is None:
        if pre_transform_kwargs is None:
            raise ValueError(
                'pre_transform_kwargs must be provided to compute norm_dict '
                '(needed to run the real pre_transform pipeline).')
        print('Computing norm_dict from training graphs...')
        norm_dict = _compute_norm(train_graphs, pre_transform_kwargs)

    _apply_norm(train_graphs, norm_dict, device)
    _apply_norm(val_graphs, norm_dict, device)

    train_loader = PyGDataLoader(
        train_graphs, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = PyGDataLoader(
        val_graphs, batch_size=eval_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, norm_dict


def prepare_test_dataloader(
    node_feats, graph_feats, feat_labels, labels, batch_size=32, num_workers=1,
    norm_dict=None, seed=0, max_graphs=None, pre_transform_kwargs=None
):
    """Prepare a test dataloader from Cartesian phase-space node features."""
    pl.seed_everything(seed)

    graphs = _build_graphs(
        node_feats, graph_feats, feat_labels, labels, max_graphs=max_graphs)

    device = graphs[0].pos.device

    if norm_dict is None:
        if pre_transform_kwargs is None:
            raise ValueError(
                'pre_transform_kwargs must be provided to compute norm_dict '
                '(needed to run the real pre_transform pipeline).')
        norm_dict = _compute_norm(graphs, pre_transform_kwargs)

    _apply_norm(graphs, norm_dict, device)

    loader = PyGDataLoader(
        graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return loader, norm_dict
