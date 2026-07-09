"""Prior box for the stellar-stream perturber-inference model.

9 free parameters, all sampled directly in physical units:

    log_mass            - log10(perturber mass / 1e7 Msun)
    log_radius          - log10(perturber scale radius [kpc])
    v_rel_perp          - relative velocity, perpendicular to stream flow [km/s]
    v_rel_para          - relative velocity, parallel to stream flow [km/s]
    angle_pos_impact    - impact position angle (alpha_position) [deg]
    angle_vel_delta     - offset between position angle and velocity angle [deg]
    impact_param        - impact parameter, in units of perturber scale radii
                          (NOT kpc - converted to kpc in sims.py)
    time_impact         - time before present of impact [Gyr]; sign-flipped
                          before use in sims.py (impact is in the past)
    phi1_impact_today   - stream longitude (AAU frame) where impact occurs today [deg]

sample_prior() applies a fixed detectability cut (delta_V > 3 km/s, see accept()),
so the resulting prior is over the accepted subset of the box,
not the full box - log_prior() accounts for this in its normalization.
"""

from collections.abc import Iterator

import numpy as np
from scipy.stats import uniform

_G_UNITS = 4.302e-6  # kpc (km/s)^2 / Msun
_DELTA_V_MIN = 3.0   # km/s, detectability cut

PARAM_NAMES = [
    "log_mass",
    "log_radius",
    "v_rel_perp",
    "v_rel_para",
    "angle_pos_impact",
    "angle_vel_delta",
    "impact_param",
    "time_impact",
    "phi1_impact_today",
]


class Prior:
    """Prior distribution for the stream perturber parameters."""

    label_ordering = PARAM_NAMES
    # name -> column index, so accept()'s callers below never depend on
    # label_ordering's actual order - only on names.
    _idx = {name: i for i, name in enumerate(label_ordering)}

    def __init__(self, seed=None):
        self.log_mass_dist = uniform(-1, np.log10(50) - (-1))
        self.log_radius_dist = uniform(-2, np.log10(2.5) - (-2))
        self.v_rel_perp_dist = uniform(loc=0, scale=200)  # km/s
        self.v_rel_para_dist = uniform(loc=-200, scale=400)  # km/s
        self.angle_pos_impact_dist = uniform(loc=0, scale=180)  # deg
        self.angle_vel_delta_dist = uniform(loc=-90, scale=180)  # deg
        self.impact_param_dist = uniform(loc=0.5, scale=4.5)  # scale radii
        self.time_impact_dist = uniform(loc=0., scale=0.45)  # Gyr ago
        self.phi1_impact_today_dist = uniform(loc=-15, scale=7)  # deg

        self.prior_min, self.prior_max = self._box_bounds()
        self.prior = uniform(self.prior_min, self.prior_max - self.prior_min)
        self._r_accept = None
        self._rng = np.random.default_rng(seed)

    def _box_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(prior_min, prior_max), read via .support() - frozen scipy
        uniform objects' .a/.b are always (0, 1) regardless of loc/scale
        (the standardized support), not the actual shifted box.
        """
        prior_min, prior_max = [], []
        for label in self.label_ordering:
            lo, hi = getattr(self, f'{label}_dist').support()
            prior_min.append(lo)
            prior_max.append(hi)
        return np.array(prior_min), np.array(prior_max)

    def _rvs(self, n_samples: int) -> np.ndarray:
        """Draw n_samples rows from the prior box.

        `self.prior`'s loc/scale are 9-length arrays (one uniform per
        parameter), so `.rvs` needs an explicit (n_samples, 9) size -
        `.rvs(size=n_samples)` alone doesn't broadcast against the
        parameters' own shape. Draws from self._rng (seeded via the
        constructor's `seed` arg), so all sampling on this instance -
        sample_prior, acceptance_rate, iter_params - is reproducible.
        """
        return self.prior.rvs(
            size=(n_samples, len(self.label_ordering)), random_state=self._rng)

    def _accept_cols(self, rows: np.ndarray) -> np.ndarray:
        """Detectability cut (accept()) applied to (..., 9) physical rows,
        looking up the 4 needed columns by name - robust to label_ordering
        being reordered.
        """
        log_mass = rows[..., self._idx['log_mass']]
        impact_param = rows[..., self._idx['impact_param']]
        v_rel_perp = rows[..., self._idx['v_rel_perp']]
        v_rel_para = rows[..., self._idx['v_rel_para']]
        return self.accept(log_mass, impact_param, v_rel_perp, v_rel_para)

    def accept(self, log_mass, impact_param, v_rel_perp, v_rel_para):
        """Detectability cut: delta_V > 3 km/s.

        mass_perturber (physical Msun) is 10**log_mass * 1e7 - the same
        conversion sims.py applies before simulating (see module docstring).
        """
        mass_perturber = 10 ** log_mass * 1e7
        v_rel_mag = np.sqrt(v_rel_perp ** 2 + v_rel_para ** 2)
        delta_v = 2 * _G_UNITS * mass_perturber / (impact_param * v_rel_mag)
        return delta_v > _DELTA_V_MIN

    def acceptance_rate(self, n_samples=10_000) -> float:
        """Estimate the delta_V > 3 km/s acceptance rate via Monte Carlo."""
        if self._r_accept is not None:
            return self._r_accept

        sample = self._rvs(n_samples)
        n_accept = np.sum(self._accept_cols(sample))

        self._r_accept = n_accept / n_samples
        return self._r_accept

    def sample_prior(
        self, n_samples, n_draw=None, n_oversample=10, n_oversample_max=1000,
    ) -> np.ndarray:
        """Rejection-sample the prior box under the delta_V > 3 km/s cut."""
        samples = []
        n_collected = 0
        n_drawn = 0
        n_draw = n_draw if n_draw is not None else n_samples * n_oversample
        while n_collected < n_samples and n_drawn < n_samples * n_oversample_max:
            candidates = self._rvs(n_draw)
            n_drawn += n_draw
            accepted = self._accept_cols(candidates)
            batch = candidates[accepted]
            samples.append(batch)
            n_collected += len(batch)

        if n_collected == 0:
            raise RuntimeError(
                f'No candidates passed the delta_V > 3 km/s cut after '
                f'{n_drawn:,} draws (requested {n_samples} samples).')
        return np.concatenate(samples)[:n_samples]

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log-density of the prior, in physical units.

        Uniform over the box, restricted to the delta_V > 3 km/s accepted
        region (accept()): sample_prior() only ever draws from that
        region, so its true density is 1 / (box_volume * acceptance_rate)
        inside it, and -inf outside (whether rejected by the box bounds or
        by the delta_V cut).

        Args:
            theta: (..., 9) physical-unit rows, ordered per label_ordering.

        Returns:
            (...,) array of log-densities, -inf outside the accepted region.
        """
        theta = np.asarray(theta)

        in_box = np.all((theta >= self.prior_min) & (theta <= self.prior_max), axis=-1)
        detectable = self._accept_cols(theta)
        valid = in_box & detectable

        box_volume = np.prod(self.prior_max - self.prior_min)
        log_density = -np.log(box_volume * self.acceptance_rate())
        return np.where(valid, log_density, -np.inf)

    def iter_params(self, n_samples: int) -> Iterator[np.ndarray]:
        """
        Lazily draw theta from the prior.

        Draws are generated one at a time instead of allocating arrays
        of length `n_samples` up front, so memory use stays flat no matter
        how large `n_samples` is.

        Args:
            n_samples: Number of galaxies to generate parameters for.

        Yields:
            theta, one per galaxy.
        """
        for _ in range(n_samples):
            theta = self.sample_prior(1)[0]
            yield theta


def default_norm_dict(n_dim=6) -> dict:
    """Generic norm_dict for the random_init debug model - no real data needed.

    Args:
        n_dim: Width of the model's per-node x features (phi1, phi2, dist,
            pm1, pm2, vr - see sims.py); a placeholder until model_io.py's
            debug configs are updated for the stream node-feature schema.

    Returns:
        norm_dict with theta_loc/theta_scale spanning the fixed prior box,
        and unit-scale x_loc/x_scale. No cond_loc/cond_scale: this
        project's model has no conditioning dimension.
    """
    default_prior = Prior()
    prior_min, prior_max = default_prior.prior_min, default_prior.prior_max

    theta_loc = (prior_max + prior_min) / 2
    theta_scale = (prior_max - prior_min) / 2
    return {
        'theta_loc': theta_loc.tolist(),
        'theta_scale': theta_scale.tolist(),
        'x_loc': [0.0] * n_dim,
        'x_scale': [1.0] * n_dim,
    }
