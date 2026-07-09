""" Simulate perturbed stellar-stream realizations (process pool). """

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="Nbody_streams")
warnings.filterwarnings("ignore", category=FutureWarning, module="Nbody_streams")

# Belt-and-suspenders alongside agama.setNumThreads() below: this
# makes single-threaded OpenMP the process-wide default read at
# each worker's first parallel region, regardless of how that
# worker process's OpenMP runtime gets (re-)initialized after fork.
# Must be set before `import agama`.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from glob import glob

import agama
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stream_sims import prior, sims


def _init_worker(sample_threads: int = 1) -> None:
    """
    Pool initializer: run once in each worker process at startup.

    Each worker here is a separate OS process with its own isolated
    agama state (including its own copy of agama's internal RNG),
    Default to 1 OpenMP thread per process; _init_worker() below can raise this
    per --sample-threads. Keep n_workers x sample_threads within your
    allocated core count to avoid oversubscription.

    Args:
        sample_threads: Number of OpenMP threads this worker
            process may use internally for agama calls.
    """
    agama.setNumThreads(sample_threads)


def save_config(
    args: argparse.Namespace, prior_obj: prior.Prior, file_path: str,
) -> None:
    """
    Save the run configuration for reproducibility.

    Args:
        args: Parsed command line arguments used for this run,
            including the resolved random seed.
        prior_obj: The Prior instance this run draws from.
        file_path: Destination path for the config JSON file.
    """
    config = vars(args).copy()
    config['prior_min'] = prior_obj.prior_min.tolist()
    config['prior_max'] = prior_obj.prior_max.tolist()
    config['timestamp'] = datetime.datetime.now().isoformat()
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace with the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Simulate the 6D stellar kinematics of stellar stream + subhalo impact.')
    parser.add_argument(
        '--n-sims', type=int, default=10000,
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
             'counts subsample the fixed stripping-time distribution.')
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
    """ Simulate perturbed stream realizations from the prior. """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is None:
        # 64-bit seed for reproducibility (max for HDF5)
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    print(f'Using seed: {args.seed}')

    prior_obj = prior.Prior(seed=args.seed)
    param_stream = prior_obj.iter_params(args.n_sims)

    theta_buffer: list[np.ndarray] = []
    feats_buffer: list[np.ndarray] = []
    n_success = 0

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
            args, prior_obj, os.path.join(args.output_dir, f'config.{config_idx:d}.json'))

    def _flush() -> None:
        nonlocal file_idx
        if not theta_buffer:
            return
        file_path = os.path.join(
            args.output_dir, f'data.{file_idx:d}.h5')
        sims.write_graph_dataset(
            file_path, np.array(theta_buffer), feats_buffer,
            prior.PARAM_NAMES, headers={'seed': args.seed})
        file_idx += 1
        theta_buffer.clear()
        feats_buffer.clear()

    # cap the number of in-flight tasks so memory use stays bounded
    # no matter how large n_sims is
    max_pending = args.max_pending or args.n_workers * 4

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.sample_threads,),
    ) as pool:
        pending = {
            pool.submit(sims.simulate_one, p, args.num_particles)
            for p in itertools.islice(param_stream, max_pending)
        }
        with tqdm(total=args.n_sims, desc='Simulating') as pbar:
            while pending:
                done, pending = wait(
                    pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pbar.update(1)
                    theta, feats = fut.result()
                    if feats is not None:
                        theta_buffer.append(theta)
                        feats_buffer.append(feats)
                        n_success += 1
                        if len(theta_buffer) >= args.sims_per_file:
                            _flush()
                    next_item = next(param_stream, None)
                    if next_item is not None:
                        pending.add(
                            pool.submit(sims.simulate_one, next_item, args.num_particles))

    _flush()

    print(f'Successful simulations: {n_success} / {args.n_sims} '
          f'({n_success / args.n_sims * 100:.1f}%)')
    print(f'Saved {file_idx} file(s) to {args.output_dir}')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Time taken: {t2 - t1:.2f} seconds')
