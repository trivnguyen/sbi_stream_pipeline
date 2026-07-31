"""Median track of the unperturbed stream, and the script that fits it.

A stream's per-star coordinates are tight functions of `phi1` -- it is a
one-dimensional structure -- and a perturber shows up as the departure
from that relation, not as the relation itself. A density model trained
on raw coordinates therefore spends most of its capacity describing a
curve that carries no information about `theta`. This module measures
that curve once and stores it, so downstream models can work in offsets
from it.

One track serves every `theta`: with the perturber mass driven to zero
the other eight parameters cannot reach the stream, which `--check`
verifies by fitting a second track at a deliberately different impact
geometry and comparing the two.

Run it as a script to produce the artifact:

    python -m stream_sims.track \\
        --output /scratch/$USER/stream_tracks/9p_AAU_622.npz

Needs the simulator's dependencies (`agama`, `Nbody_streams`), so run it
where those import -- a login node is fine, it is two simulator calls.

The `.npz` holds the piecewise-cubic coefficients from scipy's
`PchipInterpolator.c` rather than the bin medians. That keeps loading
independent of scipy reproducing the same spline, and makes evaluation
downstream a `searchsorted` plus a Horner step -- cheap enough to run per
batch on a GPU (~0.4 ms for 8e5 stars).

The consumer applies

    Delta y = y - track(phi1)

which is a shear with unit Jacobian determinant, so it needs no
log-density correction. It also commutes with the measurement model as
long as `phi1` carries no uncertainty: mu(phi1) is then the same before
and after noise is added, and the recorded sigmas are unchanged. **If
`phi1` ever gains an uncertainty, that stops being true** and the track
has to be applied after noising, using the observed `phi1`.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import sims
from .prior import PARAM_NAMES

# Every simulator feature except phi1, which is the coordinate the track
# is indexed by rather than one of the tracked coordinates.
DEFAULT_COORDS = tuple(
    name for name in sims.NODE_FEATURE_NAMES if name != 'phi1')

# Six orders of magnitude below the prior's lower edge (log_mass = -1),
# i.e. 1e2 Msun: present in the simulator call, absent from the stream.
UNPERTURBED_LOG_MASS = -5.0

# The impact geometry is irrelevant at zero mass, so these are only here
# to make the call well-formed. `--check` re-fits at a different one.
BASE_GEOMETRY = dict(
    log_radius=np.log10(0.3), v_rel_perp=35.0, v_rel_para=10.0,
    angle_pos_impact=0.0, angle_vel_delta=45.0, impact_param=1.0,
    time_impact=0.25, phi1_impact_today=-12.0,
)
CHECK_GEOMETRY = dict(
    log_radius=np.log10(1.5), v_rel_perp=150.0, v_rel_para=-80.0,
    angle_pos_impact=120.0, angle_vel_delta=-60.0, impact_param=4.0,
    time_impact=0.05, phi1_impact_today=-9.0,
)


class StreamTrack:
    """Piecewise-cubic track of an unperturbed stream against `phi1`.

    Attributes:
        knots: (n_bins,) `phi1` of each bin centre.
        coef: (n_coords, 4, n_bins - 1) power-basis coefficients about
            `knots[:-1]`, as produced by `PchipInterpolator.c`.
        end_value, end_slope: (n_coords, 2) value and slope at the first
            and last knot, for the linear continuation beyond them.
        coords: Tracked coordinate names, indexing the first axis above.
        width: (n_coords, n_bins) robust scatter (1.4826 x MAD) of the
            unperturbed stream in each bin. Recorded because it can only
            be recovered by re-running the simulator; nothing here uses
            it, and the consumer's normalization is its own business.
        meta: Provenance -- theta, particle count, bin count, and a hash
            of the unperturbed stream, so a regenerated dataset cannot
            silently pair with a stale track.
    """

    def __init__(self, knots, coef, end_value, end_slope, coords,
                 width=None, meta=None):
        self.knots = np.asarray(knots, dtype=float)
        self.coef = np.asarray(coef, dtype=float)
        self.end_value = np.asarray(end_value, dtype=float)
        self.end_slope = np.asarray(end_slope, dtype=float)
        self.coords = list(coords)
        self.width = None if width is None else np.asarray(width, dtype=float)
        self.meta = dict(meta or {})

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(cls, feats, feature_names=None, coords=DEFAULT_COORDS,
            n_bins=40, meta=None):
        """Fit a track to an unperturbed stream.

        Bins are **equal-count** rather than equal-width: the stream's
        linear density varies by more than an order of magnitude along
        it, and the ends, where the track curves most, are exactly where
        equal-width bins run out of stars. Each bin contributes a median
        and a robust width, and the medians are interpolated with a
        shape-preserving `PchipInterpolator` -- a natural cubic spline
        rings at the sharp density edges, and ringing in the track would
        read as signal in the offsets.

        Args:
            feats: (n, n_features) unperturbed stream.
            feature_names: Column names of `feats`. Defaults to the
                simulator's own `NODE_FEATURE_NAMES`.
            coords: Coordinates to track (everything except `phi1`).
            n_bins: Number of equal-count `phi1` bins.
            meta: Extra provenance to record.

        Returns:
            A fitted `StreamTrack`.

        Raises:
            ValueError: If a bin holds too few stars to take a median of.
        """
        feature_names = list(feature_names or sims.NODE_FEATURE_NAMES)
        column = {name: i for i, name in enumerate(feature_names)}

        order = np.argsort(feats[:, column['phi1']])
        feats = feats[order]
        phi1 = feats[:, column['phi1']]

        edges = np.quantile(phi1, np.linspace(0, 1, n_bins + 1))
        which = np.clip(
            np.searchsorted(edges, phi1, 'right') - 1, 0, n_bins - 1)
        counts = np.bincount(which, minlength=n_bins)
        if counts.min() < 5:
            raise ValueError(
                f'thinnest phi1 bin holds {counts.min()} stars; lower '
                f'n_bins (currently {n_bins}) or simulate more particles')

        centre = np.array(
            [np.median(phi1[which == b]) for b in range(n_bins)])

        coef, end_value, end_slope, width = [], [], [], []
        for name in coords:
            values = feats[:, column[name]]
            median = np.array(
                [np.median(values[which == b]) for b in range(n_bins)])
            mad = np.array([np.median(np.abs(values[which == b] - m))
                            for b, m in zip(range(n_bins), median)])

            spline = PchipInterpolator(centre, median)
            slope = spline.derivative()
            coef.append(spline.c)
            end_value.append([spline(centre[0]), spline(centre[-1])])
            end_slope.append([slope(centre[0]), slope(centre[-1])])
            width.append(1.4826 * mad)

        return cls(
            knots=centre, coef=np.stack(coef),
            end_value=np.array(end_value), end_slope=np.array(end_slope),
            coords=coords, width=np.stack(width),
            meta=dict(meta or {}, n_bins=n_bins, n_stars=int(len(phi1)),
                      stars_per_bin=int(counts.min())))

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, name, phi1):
        """Track value of `name` at `phi1`.

        The track spans bin *centres*, so a couple of percent of stars
        fall beyond its ends. Out there it continues along its endpoint
        slope. Holding it constant instead is the obvious choice and the
        wrong one: the coordinates are still curving, and a flat track
        turns that curvature into a fake offset -- in the unperturbed
        stream as much as in a perturbed one, which is precisely where
        the offsets are supposed to vanish.
        """
        index = self.coords.index(name)
        phi1 = np.asarray(phi1, dtype=float)
        lo, hi = self.knots[0], self.knots[-1]

        clamped = np.clip(phi1, lo, hi)
        interval = np.clip(
            np.searchsorted(self.knots, clamped, 'right') - 1,
            0, self.knots.size - 2)
        dx = clamped - self.knots[interval]
        c = self.coef[index][:, interval]
        value = ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]

        value = np.where(
            phi1 < lo,
            self.end_value[index, 0] + self.end_slope[index, 0] * (phi1 - lo),
            value)
        return np.where(
            phi1 > hi,
            self.end_value[index, 1] + self.end_slope[index, 1] * (phi1 - hi),
            value)

    def project(self, feats, feature_names=None):
        """Replace each tracked coordinate by its offset from the track."""
        feature_names = list(feature_names or sims.NODE_FEATURE_NAMES)
        column = {name: i for i, name in enumerate(feature_names)}
        out = np.asarray(feats, dtype=float).copy()
        phi1 = out[:, column['phi1']]
        for name in self.coords:
            out[:, column[name]] -= self.evaluate(name, phi1)
        return out

    def unproject(self, feats, feature_names=None):
        """Inverse of `project`, using each row's own `phi1`."""
        feature_names = list(feature_names or sims.NODE_FEATURE_NAMES)
        column = {name: i for i, name in enumerate(feature_names)}
        out = np.asarray(feats, dtype=float).copy()
        phi1 = out[:, column['phi1']]
        for name in self.coords:
            out[:, column[name]] += self.evaluate(name, phi1)
        return out

    # -- storage ------------------------------------------------------------

    def save(self, path):
        """Write the track to a `.npz`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, knots=self.knots, coef=self.coef,
            end_value=self.end_value, end_slope=self.end_slope,
            width=self.width, coords=np.array(self.coords, dtype='U16'),
            meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        """Read a track written by `save`."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                knots=data['knots'], coef=data['coef'],
                end_value=data['end_value'], end_slope=data['end_slope'],
                coords=[str(name) for name in data['coords']],
                width=data['width'],
                meta=json.loads(str(data['meta'])))


# -- simulation -------------------------------------------------------------

def simulate_unperturbed(num_particles=5000, log_mass=UNPERTURBED_LOG_MASS,
                         geometry=None, seed=None):
    """Simulate one stream with the perturber mass driven to zero.

    Args:
        num_particles: Particles to simulate; capped by the fixed
            snapshot's own count.
        log_mass: log10(perturber mass / 1e7 Msun). The default is far
            below the prior, so the perturber is present in the call and
            absent from the result.
        geometry: The other eight parameters. Irrelevant at zero mass;
            defaults to `BASE_GEOMETRY`.
        seed: Randomizes release times and spray offsets. `None` keeps
            the simulator deterministic, which is what a reference track
            wants.

    Returns:
        (theta, feats): the parameter row used and the (n, 6) stream in
        `sims.NODE_FEATURE_NAMES` order.

    Raises:
        RuntimeError: If the simulator rejected these parameters.
    """
    params = dict(geometry or BASE_GEOMETRY, log_mass=log_mass)
    theta = np.array([params[name] for name in PARAM_NAMES], dtype=np.float64)

    _, feats = sims.simulate_one(theta, num_particles=num_particles, seed=seed)
    if feats is None:
        raise RuntimeError(f'simulator rejected theta={theta}')
    return theta, feats


def _stream_hash(feats):
    """Content hash of a stream, to tie a track to its source snapshot."""
    return hashlib.sha256(
        np.ascontiguousarray(feats, dtype=np.float64)).hexdigest()[:16]


# -- script -----------------------------------------------------------------

def _report_self_consistency(track, feats):
    """Print the offsets of the stream the track was fitted to."""
    offsets = track.project(feats)
    column = {name: i for i, name in enumerate(sims.NODE_FEATURE_NAMES)}
    phi1 = feats[:, column['phi1']]
    outside = (phi1 < track.knots[0]) | (phi1 > track.knots[-1])

    print(f'\n  track spans phi1 [{track.knots[0]:.1f}, '
          f'{track.knots[-1]:.1f}]; stream reaches '
          f'[{phi1.min():.1f}, {phi1.max():.1f}]')
    print(f'  {100 * outside.mean():.2f}% of stars use the linear '
          f'continuation')
    print(f'\n  {"coord":>6} {"mean |offset|":>16} {"beyond the knots":>18}')
    for name in track.coords:
        residual = np.abs(offsets[:, column[name]])
        print(f'  {name:>6} {residual[~outside].mean():>16.5f} '
              f'{residual[outside].mean():>18.5f}')


def _report_geometry_check(track, num_particles):
    """Re-fit at a different impact geometry and compare the tracks.

    The whole approach rests on one track serving every `theta`. At zero
    mass the other eight parameters cannot reach the stream, so this
    should come out at a small fraction of the stream's own width.
    """
    print('\n[check] Re-fitting at a different impact geometry...')
    _, feats = simulate_unperturbed(
        num_particles=num_particles, geometry=CHECK_GEOMETRY)
    other = StreamTrack.fit(feats, coords=track.coords,
                            n_bins=track.meta['n_bins'])

    probe = np.linspace(track.knots[0], track.knots[-1], 400)
    print(f'  {"coord":>6} {"max difference":>18} {"/ stream width":>16}')
    worst = 0.0
    for index, name in enumerate(track.coords):
        difference = np.abs(track.evaluate(name, probe)
                            - other.evaluate(name, probe))
        width = np.interp(probe, track.knots, track.width[index])
        relative = (difference / width).max()
        worst = max(worst, relative)
        print(f'  {name:>6} {difference.max():>18.5f} {relative:>16.3f}')

    verdict = 'consistent' if worst < 0.25 else 'INCONSISTENT'
    print(f'  worst disagreement: {worst:.3f} stream widths -- {verdict}')


def main():
    parser = argparse.ArgumentParser(
        description='Fit the unperturbed stream track and store it.')
    parser.add_argument(
        '--output', required=True, type=Path,
        help='Destination .npz.')
    parser.add_argument(
        '--n-bins', type=int, default=40,
        help='Equal-count phi1 bins (default: 40).')
    parser.add_argument(
        '--num-particles', type=int, default=5000,
        help='Particles to simulate (default: 5000, the snapshot max).')
    parser.add_argument(
        '--log-mass', type=float, default=UNPERTURBED_LOG_MASS,
        help='log10(mass / 1e7 Msun) standing in for no perturber '
             f'(default: {UNPERTURBED_LOG_MASS}).')
    parser.add_argument(
        '--coords', nargs='+', default=list(DEFAULT_COORDS),
        help=f'Coordinates to track (default: {" ".join(DEFAULT_COORDS)}).')
    parser.add_argument(
        '--no-check', action='store_true',
        help='Skip the second simulation that verifies one track serves '
             'every theta.')
    args = parser.parse_args()

    print(f'[sim] Simulating {args.num_particles:,} particles at '
          f'log_mass={args.log_mass}...')
    theta, feats = simulate_unperturbed(
        num_particles=args.num_particles, log_mass=args.log_mass)

    print(f'[fit] {args.n_bins} equal-count phi1 bins over '
          f'{len(feats):,} stars, tracking {", ".join(args.coords)}')
    track = StreamTrack.fit(
        feats, coords=args.coords, n_bins=args.n_bins,
        meta={
            'theta': theta.tolist(),
            'param_names': list(PARAM_NAMES),
            'feature_names': list(sims.NODE_FEATURE_NAMES),
            'num_particles': args.num_particles,
            'log_mass': args.log_mass,
            'stream_sha256': _stream_hash(feats),
        })

    _report_self_consistency(track, feats)
    if not args.no_check:
        _report_geometry_check(track, args.num_particles)

    path = track.save(args.output)
    print(f'\n[out] Wrote {path} ({path.stat().st_size / 1024:.1f} KiB)')
    print(f'[out] stream_sha256 = {track.meta["stream_sha256"]}')


if __name__ == '__main__':
    main()
