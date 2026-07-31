""" Simulate stream realizations restricted to a phi1/phi2 window.

Same simulator, prior and output layout as simulate_process.py; the only
difference is that each realization's particles are cut to a window in
the AAU stream frame before being written, standing in for the footprint
of a survey that does not see the whole stream.

The window lives in PHI1_RANGE / PHI2_RANGE below (placeholders - fill
them in) and can be overridden per run with --phi1-min etc. Any bound
left unset is unbounded, so an all-unset window (or --no-cut) writes the
full stream and this script reduces to simulate_process.py.

Usage
-----
python simulate_process_cut.py --n-sims 1000 --output-dir ./simdata_cut \\
    --phi1-min -20 --phi1-max 10 --phi2-max 1.5
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from glob import glob

warnings.filterwarnings(
    "ignore", category=UserWarning, module="Nbody_streams")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="Nbody_streams")

# Must be set before `import agama`; see simulate_process.py.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import agama
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate_process import save_config, seed_stream
from stream_sims import prior, sims

# Placeholder observing window, degrees in the AAU stream frame - fill
# in. None means unbounded on that side.
PHI1_RANGE: tuple[float | None, float | None] = (None, None)
PHI2_RANGE: tuple[float | None, float | None] = (None, None)


def _init_worker(sample_threads: int = 1) -> None:
    """
    Pool initializer: run once in each worker process at startup.

    Args:
        sample_threads: Number of OpenMP threads this worker process may
            use internally for agama calls.
    """
    agama.setNumThreads(sample_threads)


def apply_cut(
    feats: np.ndarray, bounds: dict[str, tuple[float | None, float | None]],
) -> np.ndarray:
    """
    Keep only the particles inside the phi1/phi2 window.

    Args:
        feats: (n_particles, 6) node features, columns
            sims.NODE_FEATURE_NAMES.
        bounds: Feature name -> (min, max) in that feature's own units;
            either end may be None for unbounded.

    Returns:
        The surviving rows of `feats`, in their original order. May be
        empty if the window contains no particles.

    Raises:
        KeyError: If `bounds` names a feature the simulator does not
            produce.
    """
    keep = np.ones(len(feats), dtype=bool)
    for name, (low, high) in bounds.items():
        column = feats[:, sims.NODE_FEATURE_NAMES.index(name)]
        if low is not None:
            keep &= column >= low
        if high is not None:
            keep &= column <= high
    return feats[keep]


def resolve_bounds(
    args: argparse.Namespace,
) -> dict[str, tuple[float | None, float | None]]:
    """
    Build the cut window from the parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Feature name -> (min, max), all-None under --no-cut.
    """
    if args.no_cut:
        return {'phi1': (None, None), 'phi2': (None, None)}
    return {
        'phi1': (args.phi1_min, args.phi1_max),
        'phi2': (args.phi2_min, args.phi2_max),
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace with the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Simulate the 6D stellar kinematics of stellar stream '
                    '+ subhalo impact, cut to a phi1/phi2 window.')
    parser.add_argument(
        '--num-sims', type=int, default=10000,
        help='Number of stream realizations to attempt to simulate.')
    parser.add_argument(
        '--output-dir', type=str, default='./simdata',
        help='Directory to create (if needed) and save output '
             'files to.')
    parser.add_argument(
        '--n-workers', type=int, default=os.cpu_count(),
        help='Number of worker processes to use.')
    parser.add_argument(
        '--sample-threads', type=int, default=1,
        help='Number of OpenMP threads each worker process may use '
             'internally for agama calls. Keep n-workers x '
             'sample-threads within your allocated core count to '
             'avoid oversubscription. Tune this per-cluster.')
    parser.add_argument(
        '--sims-per-file', type=int, default=1000,
        help='Maximum number of successful simulations stored in each '
             'output HDF5 file.')
    parser.add_argument(
        '--num-particles', type=int, default=None,
        help='Number of stream particles to simulate per realization '
             '(default: the fixed snapshot count, stream_sims.sims.META '
             '["num_particles"]). Must not exceed the default; smaller '
             'counts subsample the fixed stripping-time distribution. '
             'Counted before the phi1/phi2 cut.')
    parser.add_argument(
        '--phi1-min', type=float, default=PHI1_RANGE[0],
        help=f'Lower phi1 bound of the window, deg (default: '
             f'{PHI1_RANGE[0]}; unset means unbounded).')
    parser.add_argument(
        '--phi1-max', type=float, default=PHI1_RANGE[1],
        help=f'Upper phi1 bound of the window, deg (default: '
             f'{PHI1_RANGE[1]}; unset means unbounded).')
    parser.add_argument(
        '--phi2-min', type=float, default=PHI2_RANGE[0],
        help=f'Lower phi2 bound of the window, deg (default: '
             f'{PHI2_RANGE[0]}; unset means unbounded).')
    parser.add_argument(
        '--phi2-max', type=float, default=PHI2_RANGE[1],
        help=f'Upper phi2 bound of the window, deg (default: '
             f'{PHI2_RANGE[1]}; unset means unbounded).')
    parser.add_argument(
        '--no-cut', action='store_true',
        help='Ignore every bound above and keep the full stream, i.e. '
             'reproduce simulate_process.py.')
    parser.add_argument(
        '--min-particles', type=int, default=1,
        help='Discard a realization if fewer than this many particles '
             'survive the cut. Such a stream carries no usable '
             'observation, and an empty one cannot be written.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility.')
    parser.add_argument(
        '--max-pending', type=int, default=None,
        help='Maximum number of in-flight simulations kept in '
             'memory at once (default: 4x n-workers). Lower this '
             'if simulations run out of memory.')
    parser.add_argument(
        '--append', action='store_true',
        help='Append to existing output files instead of overwriting '
             'them. If set, the output directory must already exist.')
    return parser.parse_args()


def main() -> None:
    """ Simulate windowed stream realizations from the prior. """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is None:
        # 64-bit seed for reproducibility (max for HDF5)
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    print(f'Using seed: {args.seed}')

    bounds = resolve_bounds(args)
    print('Cut window (deg): ' + ', '.join(
        f'{name} [{low}, {high}]' for name, (low, high) in bounds.items()))

    # Two independent child streams off the one resolved seed: the prior
    # draws, and the simulator's per-realization seeds. See
    # simulate_process.py.
    prior_seq, sim_seq = np.random.SeedSequence(args.seed).spawn(2)

    prior_obj = prior.Prior(seed=prior_seq)
    task_stream = zip(prior_obj.iter_params(args.num_sims), seed_stream(sim_seq))

    theta_buffer: list[np.ndarray] = []
    feats_buffer: list[np.ndarray] = []
    n_success = 0
    n_emptied = 0

    # append mode: find the next available file index to avoid overwriting
    if not args.append:
        file_idx = 0
        save_config(
            args, prior_obj, os.path.join(args.output_dir, 'config.0.json'))
    else:
        if args.output_dir is None or not os.path.exists(args.output_dir):
            raise ValueError(
                f'Output directory {args.output_dir} does not exist, '
                'cannot append.')
        existing_files = glob(os.path.join(args.output_dir, 'data.*.h5'))
        existing_indices = [
            int(f.split('.')[1]) for f in existing_files
            if f.split('.')[1].isdigit()
        ]
        file_idx = max(existing_indices, default=-1) + 1

        existing_configs = glob(os.path.join(args.output_dir, 'config.*.json'))
        existing_config_indices = [
            int(f.split('.')[1]) for f in existing_configs
            if f.split('.')[1].isdigit()
        ]
        config_idx = max(existing_config_indices, default=-1) + 1
        save_config(
            args, prior_obj,
            os.path.join(args.output_dir, f'config.{config_idx:d}.json'))

    # nan rather than None for the unbounded ends: HDF5 attributes have
    # no null, and every bound stays present in the file either way.
    cut_headers = {
        f'{name}_{edge}': float('nan') if value is None else float(value)
        for name, (low, high) in bounds.items()
        for edge, value in (('min', low), ('max', high))
    }

    def _flush() -> None:
        nonlocal file_idx
        if not theta_buffer:
            return
        file_path = os.path.join(
            args.output_dir, f'data.{file_idx:d}.h5')
        sims.write_graph_dataset(
            file_path, np.array(theta_buffer), feats_buffer,
            prior.PARAM_NAMES, headers={'seed': args.seed, **cut_headers})
        file_idx += 1
        theta_buffer.clear()
        feats_buffer.clear()

    # cap the number of in-flight tasks so memory use stays bounded
    # no matter how large num_sims is
    max_pending = args.max_pending or args.n_workers * 4

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.sample_threads,),
    ) as pool:
        pending = {
            pool.submit(sims.simulate_one, p, args.num_particles, seed=s)
            for p, s in itertools.islice(task_stream, max_pending)
        }
        with tqdm(total=args.num_sims, desc='Simulating') as pbar:
            while pending:
                done, pending = wait(
                    pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pbar.update(1)
                    theta, feats = fut.result()
                    if feats is not None:
                        # Cut in the parent, not the worker: the arrays are
                        # small next to the simulation itself, and this keeps
                        # the simulator call identical to simulate_process.py.
                        feats = apply_cut(feats, bounds)
                        if len(feats) < args.min_particles:
                            n_emptied += 1
                        else:
                            theta_buffer.append(theta)
                            feats_buffer.append(feats)
                            n_success += 1
                            if len(theta_buffer) >= args.sims_per_file:
                                _flush()
                    next_item = next(task_stream, None)
                    if next_item is not None:
                        next_theta, next_seed = next_item
                        pending.add(pool.submit(
                            sims.simulate_one, next_theta,
                            args.num_particles, seed=next_seed))

    _flush()

    print(f'Successful simulations: {n_success} / {args.num_sims} '
          f'({n_success / args.num_sims * 100:.1f}%)')
    print(f'Dropped by the cut (< {args.min_particles} particles left): '
          f'{n_emptied}')
    print(f'Saved {file_idx} file(s) to {args.output_dir}')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Time taken: {t2 - t1:.2f} seconds')
