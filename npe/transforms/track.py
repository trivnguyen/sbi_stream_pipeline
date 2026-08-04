"""Unperturbed stream track, as a PyG graph transform.

Every per-star coordinate is a tight function of `phi1` -- the stream is
a one-dimensional structure -- and the perturber signal is the departure
from that relation. Subtracting the track leaves the GNN reading only the
departure.

The `.npz` is fitted once by `stream_sims.track` and shared verbatim with
the NLE pipeline, so both estimators see the same reference stream and
any systematic in it is common to them.

What NPE does *not* need
------------------------
For NLE the same projection had to be volume preserving and invertible,
because `x` is the modelled variable there: the density picks up a
log-Jacobian and `sample` has to map draws back. Here `x` is only the
conditioning input to the embedding network -- the density is over
`theta` -- so no Jacobian and no inverse ever enter. The projection could
be non-invertible without consequence; it simply happens not to be.

`pos` as well as `x`
--------------------
`pos = (phi1, phi2)` is what the adaptive-kNN graph is built from, and
`x` is the node feature vector; `phi1` and `phi2` appear in both.
Projecting `pos` too straightens the stream before neighbours are chosen,
so edges run along the track rather than cutting across its bend. That
also shrinks the `phi2` extent by roughly an order of magnitude, which
makes neighbour selection more strongly `phi1`-ordered -- a real change
to the graph, not just to the features, so `project_pos` is a knob rather
than a constant.
"""

import json

import numpy as np
import torch


def load_track_npz(path):
    """Read a `.npz` written by `stream_sims.track`.

    Read directly rather than through `stream_sims.track.StreamTrack`:
    importing that package pulls in `agama` and `Nbody_streams`, which a
    training job has no use for. The file is six arrays and a JSON
    string, and `meta['stream_sha256']` is what actually ties a track to
    the stream it was fitted from.
    """
    with np.load(path, allow_pickle=False) as data:
        return {
            'knots': data['knots'],
            'coef': data['coef'],
            'end_value': data['end_value'],
            'end_slope': data['end_slope'],
            'coords': [str(name) for name in data['coords']],
            'meta': json.loads(str(data['meta'])),
        }


class TrackProjection:
    """Re-express node features as offsets from the unperturbed track.

    Args:
        path: `.npz` written by `stream_sims.track`.
        feat_labels: Column order of `data.x`.
        project_pos: Also project `data.pos`, which the graph is built
            from. `pos` is assumed to be `(phi1, phi2)`, as
            `datasets.cartesian._build_graphs` writes it.

    Raises:
        ValueError: If `phi1` is missing from `feat_labels`, if the track
            does not cover every other modelled coordinate, or if
            `project_pos` is set and the track has no `phi2`.
    """

    def __init__(self, path, feat_labels, project_pos=True):
        fitted = load_track_npz(path)
        feat_labels = list(feat_labels)
        if 'phi1' not in feat_labels:
            raise ValueError(
                "feat_labels has no 'phi1'; the track is indexed by it")

        fitted_coords = fitted['coords']
        missing = [name for name in feat_labels
                   if name != 'phi1' and name not in fitted_coords]
        if missing:
            raise ValueError(
                f'track has no entry for {missing}; it was fitted on '
                f'{fitted_coords}')

        rows = [fitted_coords.index(name) for name in feat_labels
                if name != 'phi1']
        self.coords = [fitted_coords[row] for row in rows]
        self.feat_labels = feat_labels
        self.path = str(path)
        self.meta = dict(fitted['meta'])
        self.project_pos = project_pos

        as_tensor = lambda v: torch.tensor(v, dtype=torch.float32)
        self.knots = as_tensor(fitted['knots'])
        self.coef = as_tensor(fitted['coef'])[rows]
        self.end_value = as_tensor(fitted['end_value'])[rows]
        self.end_slope = as_tensor(fitted['end_slope'])[rows]
        self.columns = torch.tensor(
            [feat_labels.index(name) for name in self.coords],
            dtype=torch.long)
        self.phi1_column = feat_labels.index('phi1')

        if project_pos:
            if 'phi2' not in self.coords:
                raise ValueError(
                    'project_pos needs a tracked phi2, since pos is '
                    '(phi1, phi2)')
            self.phi2_row = self.coords.index('phi2')

    def __repr__(self) -> str:
        return (f'TrackProjection(coords={self.coords}, '
                f'bins={self.knots.numel()}, project_pos={self.project_pos})')

    def evaluate(self, phi1: torch.Tensor) -> torch.Tensor:
        """Track values at `phi1`, as (n, n_coords) in `self.coords` order.

        Beyond the knots the track continues along its endpoint slope.
        Holding it constant instead would turn the coordinates' real
        curvature into a fake offset.
        """
        knots = self.knots.to(phi1.device)
        low, high = knots[0], knots[-1]
        clamped = phi1.clamp(low, high)
        interval = (torch.searchsorted(knots, clamped.contiguous(),
                                       right=True) - 1
                    ).clamp(0, knots.numel() - 2)

        delta = (clamped - knots[interval]).unsqueeze(-1)
        c = self.coef.to(phi1.device)[:, :, interval].permute(2, 0, 1)
        value = (((c[..., 0] * delta + c[..., 1]) * delta + c[..., 2])
                 * delta + c[..., 3])

        end_value = self.end_value.to(phi1.device)
        end_slope = self.end_slope.to(phi1.device)
        value = torch.where(
            (phi1 < low).unsqueeze(-1),
            end_value[:, 0] + end_slope[:, 0] * (phi1 - low).unsqueeze(-1),
            value)
        return torch.where(
            (phi1 > high).unsqueeze(-1),
            end_value[:, 1] + end_slope[:, 1] * (phi1 - high).unsqueeze(-1),
            value)

    def __call__(self, batch):
        """Project `batch.x` (and optionally `batch.pos`) in place of a copy.

        Node-wise, so a `Batch` of many graphs needs no special handling.
        `phi1` is read before anything is written, and is never itself
        projected, so the order of the two writes does not matter.
        """
        batch = batch.clone()
        if batch.x.shape[1] < len(self.feat_labels):
            raise ValueError(
                f'x has {batch.x.shape[1]} columns, expected at least '
                f'{len(self.feat_labels)} ({self.feat_labels})')

        values = self.evaluate(batch.x[:, self.phi1_column])

        x = batch.x.clone()
        x[:, self.columns.to(x.device)] -= values
        batch.x = x

        if self.project_pos:
            pos = batch.pos.clone()
            pos[:, 1] -= values[:, self.phi2_row]
            batch.pos = pos

        return batch
