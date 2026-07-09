"""Simulate perturbed stellar-stream realizations (Agama + Nbody_streams).

Batch runner + HDF5 writer for simulate_round.py, matching the dSph
sibling project's sims.py structure (module-level fixed setup, a
ProcessPoolExecutor-based batch runner, an HDF5 graph-dataset writer).
The physical-space conversions implied by tsnpe.prior's 9 free parameters
(angle_vel_at_impact, impact_parameter in kpc) happen here, in
simulate_one - not in prior.py or proposal.py, since they're the
simulator's own inputs.
"""

import os

# Must be set before `import agama`, so every worker process defaults to
# single-threaded OpenMP regardless of fork timing.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import agama
import h5py
import numpy as np
from tqdm import tqdm

import Nbody_streams.nbody_streams as nbody

from . import sims_utils

agama.setUnits(mass=1, length=1, velocity=1)
agama.setNumThreads(1)

POTENTIALS_DIR = Path(__file__).resolve().parent / 'potentials'

# Rotation matrix ICRS -> AAU stream frame (Shipp+2019).
R_AAU = np.array([
    [0.83697865, 0.29481904, -0.4610298],
    [0.51616778, -0.70514011, 0.4861566],
    [0.18176238, 0.64487142, 0.74236331],
])
NODE_FEATURE_NAMES = ['phi1', 'phi2', 'dist', 'pm1', 'pm2', 'vr']


def _load_fixed_data():
    """Load the fixed MW+LMC host potential, unperturbed stream snapshot,
    progenitor metadata, and per-orbit-point stripping-time distribution.

    Loaded once at import time, not lazily: ProcessPoolExecutor workers fork
    from the parent process
    """
    potMW = agama.Potential(file=str(POTENTIALS_DIR / 'McMillan17_nora.ini'))
    accMW = np.loadtxt(POTENTIALS_DIR / 'accMW')
    trajLMC = np.loadtxt(POTENTIALS_DIR / 'trajLMC')
    potacc = agama.Potential(type='UniformAcceleration', file=accMW)
    potLMC = agama.Potential(file=str(POTENTIALS_DIR / 'LMC_nora.ini'))
    potLMCm = agama.Potential(potential=potLMC, center=trajLMC)
    pot_total = agama.Potential(potMW, potLMCm, potacc)

    with open(POTENTIALS_DIR / 'stream_unperturbed_622.pkl', 'rb') as f:
        stream_data = pickle.load(f)
    stream_unperturb = stream_data['stream']
    stream_unperturb['phi1'] = stream_data['phi1']
    stream_unperturb['phi2'] = stream_data['phi2']
    meta = stream_data['metadata']

    distrib_stripping = np.load(POTENTIALS_DIR / 'distrib_stripping_622.npy')
    return pot_total, stream_unperturb, meta, distrib_stripping


POT_TOTAL, STREAM_UNPERTURB, META, DISTRIB_STRIPPING = _load_fixed_data()


def _init_worker(sample_threads: int) -> None:
    """Pool initializer: run once in each worker process at startup."""
    agama.setNumThreads(sample_threads)


def _time_stripping_for(num_particles: int) -> np.ndarray:
    """time_stripping array for `num_particles`.

    `create_particle_spray_stream` requires len(time_stripping) ==
    num_particles // 2 + 1 exactly. DISTRIB_STRIPPING is fixed at
    META['num_particles'] particles' worth; a smaller num_particles
    subsamples it (evenly spaced indices) down to the required length -
    num_particles can't exceed META['num_particles'], since there's
    nothing to subsample from beyond the fixed array.

    Args:
        num_particles: Requested particle count.

    Returns:
        time_stripping array of length num_particles // 2 + 1.
    """
    if num_particles == META['num_particles']:
        return DISTRIB_STRIPPING
    if num_particles > META['num_particles']:
        raise ValueError(
            f"num_particles={num_particles} exceeds the fixed snapshot's "
            f"META['num_particles']={META['num_particles']}; "
            'DISTRIB_STRIPPING has no more points to subsample from.')
    target_len = num_particles // 2 + 1
    idx = np.linspace(0, len(DISTRIB_STRIPPING) - 1, target_len).round().astype(int)
    return DISTRIB_STRIPPING[idx]


def simulate_one(
    theta: np.ndarray,
    num_particles: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Simulate one perturbed stream realization.

    Runs in a worker process. Defined at module level so it can be
    pickled and sent to worker processes by `ProcessPoolExecutor`.

    Args:
        theta: Length-9 physical-unit row, ordered per [log_mass, log_radius, v_perp,
            v_para, angle_pos, angle_delta, impact_param, time, phi1].
        num_particles: Number of stream particles to simulate. Defaults to
            the fixed snapshot's META['num_particles'] if not given; see
            `_time_stripping_for` for how smaller counts are handled.

    Returns:
        (theta, feats) if accepted, or (None, None) if the simulation
        failed - feats is an (n_particles, 6) array, columns
        NODE_FEATURE_NAMES (phi1, phi2, dist, pm1, pm2, vr).
    """
    (log_mass, log_radius, v_perp, v_para, angle_pos, angle_delta,
     impact_param, time_before_present, phi1_impact) = theta

    # Derived physical-space conversions
    mass_perturber = 10 ** log_mass * 1e7  # Msun
    scaleradius_perturber = 10 ** log_radius  # kpc
    angle_vel_at_impact = angle_pos + angle_delta  # deg
    impact_parameter_kpc = impact_param * scaleradius_perturber  # kpc
    time_impact = -time_before_present  # Gyr, sign-flipped (impact is in the past)
    delta_phi1 = 0.5  # fixed
    time_window = sims_utils.compute_time_window(v_perp, v_para, impact_parameter_kpc)
    num_particles = num_particles if num_particles is not None else META['num_particles']
    time_stripping = _time_stripping_for(num_particles)

    try:
        pert_dict = sims_utils.create_perturber_dict(
            STREAM_UNPERTURB['part_xv'], STREAM_UNPERTURB['phi1'], POT_TOTAL,
            mass_perturber, scaleradius_perturber,
            phi1_impact, time_impact, impact_parameter_kpc,
            v_para, v_perp,
            alpha_position=angle_pos, alpha_velocity=angle_vel_at_impact,
            delta_phi1=delta_phi1, time_window=time_window,
        )
        stream_perturb = nbody.fast_sims.create_particle_spray_stream(
            pot_host=POT_TOTAL,
            initmass=META['prog_mass_Msun'],
            scaleradius=META['prog_scaleradius_kpc'],
            prog_pot_kind='Plummer',
            sat_cen_present=META['prog_wtoday'],
            num_particles=num_particles,
            time_end=0.0,
            time_total=META['Age_stream_Gyr'],
            save_rate=1,
            time_stripping=time_stripping,
            add_perturber=pert_dict,
            verbose=False,
            dissolve_progenitor=True,
        )
        coords = sims_utils.galcen_to_stream_coords(
            stream_perturb['part_xv'], R_AAU)
    except Exception:
        return None, None

    feats = np.column_stack([coords[name] for name in NODE_FEATURE_NAMES])
    valid = np.isfinite(feats).all(axis=1)
    if valid.sum() == 0:
        return None, None
    return theta, feats[valid]


def run_simulation_batch(
    theta: np.ndarray,
    n_jobs: int = 0,
    use_multiprocessing: bool = True,
    sample_threads: int = 1,
    num_particles: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate a batch of stream realizations, keeping only successes.

    Args:
        theta: (N, 9) array of physical-unit parameters, one row per
            simulation (see `simulate_one` for column order).
        n_jobs: Worker processes to use (0 -> os.cpu_count()). Ignored if
            `use_multiprocessing` is False.
        use_multiprocessing: If False, simulate serially in this process
            (useful for debugging).
        sample_threads: OpenMP threads each worker process may use
            internally for agama calls.
        num_particles: Number of stream particles per simulation; see
            `simulate_one`.

    Returns:
        Tuple of (theta, feats_list): `theta` is the (n_success, 9)
        subset of the input that succeeded, and `feats_list` is the
        matching list of (n_i, 6) node-feature arrays (columns
        NODE_FEATURE_NAMES).
    """
    theta = np.asarray(theta)
    theta_list, feats_list = [], []

    if not use_multiprocessing:
        for row in tqdm(theta, total=len(theta), desc='Simulating'):
            row_out, feats = simulate_one(row, num_particles)
            if feats is not None:
                theta_list.append(row_out)
                feats_list.append(feats)
        return np.array(theta_list), feats_list

    n_workers = n_jobs or os.cpu_count()
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(sample_threads,),
    ) as pool:
        futures = [pool.submit(simulate_one, row, num_particles) for row in theta]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Simulating'):
            row_out, feats = fut.result()
            if feats is not None:
                theta_list.append(row_out)
                feats_list.append(feats)

    return np.array(theta_list), feats_list


def write_graph_dataset(
    path: str,
    theta: np.ndarray,
    feats_list: list[np.ndarray],
    param_names: list[str],
    headers: dict | None = None,
) -> None:
    """Write a batch of simulated streams to an HDF5 graph dataset.

    Layout matches what `jgnn.datasets.io.read_graph_dataset` expects:
    node features (one flat dataset per name in NODE_FEATURE_NAMES,
    each galaxy's particles concatenated together, split back out via
    'ptr'), and one graph-level dataset per entry in `param_names`.

    Args:
        path: Destination HDF5 path.
        theta: (n_sims, len(param_names)) physical-unit parameters.
        feats_list: Length-n_sims list of (n_i, 6) node-feature arrays,
            columns NODE_FEATURE_NAMES (phi1, phi2, dist, pm1, pm2, vr).
        param_names: Names for `theta`'s columns, in order.
        headers: Extra scalar attributes to store (e.g. round, tau).
    """
    theta = np.asarray(theta)
    n_particles = np.array([feats.shape[0] for feats in feats_list])
    ptr = np.cumsum([0] + n_particles)
    stacked = np.concatenate(feats_list, axis=0)

    graph_features = list(param_names) + ['num_particles', 'ptr']
    all_features = NODE_FEATURE_NAMES + graph_features

    with h5py.File(path, 'w') as f:
        for i, name in enumerate(NODE_FEATURE_NAMES):
            f.create_dataset(name, data=stacked[:, i])
        f.create_dataset('num_particles', data=n_particles)
        f.create_dataset('ptr', data=ptr)
        for i, name in enumerate(param_names):
            f.create_dataset(name, data=theta[:, i])

        f.attrs['all_features'] = all_features
        f.attrs['node_features'] = NODE_FEATURE_NAMES
        f.attrs['graph_features'] = graph_features
        for key, value in (headers or {}).items():
            f.attrs[key] = value
