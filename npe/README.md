# NPE pipeline

Amortized Neural Posterior Estimation on top of `jgnn`: simulate wide-prior
training data with the shared `stream_sims` simulator, train a
GNN-embedding + normalizing-flow model over the 9 perturber parameters,
checkpoint it for `tsnpe/` to fine-tune against a real target.

## Layout

```
simulate_process.py   simulate training data (ProcessPoolExecutor; each
                       worker is a separate OS process with its own
                       isolated agama state - no locking needed). Draws
                       theta from stream_sims.prior.Prior, simulates each
                       row with stream_sims.sims.simulate_one.
simulate_process_cut.py  the same, with each realization's particles cut
                       to a phi1/phi2 window (a survey footprint)
train_npe.py           train the model
configs/
  chebconv_9params.py   9-param perturber prior, AAU-frame node features
                         (phi1, phi2, dist, pm1, pm2, vr)
slurm/
  submit.sh              submit train_npe.py to SLURM with per-run
                          log bookkeeping
  train_npe.sbatch        the actual job script (usually launched via
                           submit.sh, not directly)
```

## Simulate training data

```bash
python simulate_process.py \
    --n-sims 100000 --n-workers 24 \
    --output-dir /scratch/$USER/stream_datasets/9p_AAU
```

Draws theta from `stream_sims.prior.Prior` (rejection-sampled under the
delta_V > 3 km/s detectability cut) and simulates each perturbed stream
realization with Agama + `Nbody_streams`, writing AAU-frame node features
(`phi1, phi2, dist, pm1, pm2, vr`, one row per surviving particle) to
sharded HDF5 files (`--sims-per-file`). `--seed` fixes the prior draws for
reproducibility (defaults to a fresh 32-bit seed, printed at startup and
recorded in `config.<i>.json`); `--append` resumes into an existing output
directory instead of overwriting it. `--max-pending` bounds in-flight
simulations so memory stays flat regardless of `--n-sims`.

`--seed` fixes the whole run, not just the prior: it also drives each
realization's release times and particle-spray offsets, so no two streams
in a dataset share a stochastic realization. Rerunning with the same
`--seed` reproduces the dataset exactly.

### Cut to an observing window

```bash
python simulate_process_cut.py \
    --n-sims 100000 --n-workers 24 \
    --output-dir /scratch/$USER/stream_datasets/9p_AAU_cut \
    --phi1-min -20 --phi1-max 5 --phi2-min -1 --phi2-max 1
```

Identical to the above, except that each realization's particles are cut
to a window in the AAU stream frame before being written. The default
window is the `PHI1_RANGE` / `PHI2_RANGE` placeholders at the top of the
script (both currently unbounded - fill them in); any bound left unset is
unbounded, and `--no-cut` disables the cut entirely, reproducing
`simulate_process.py` byte for byte at the same `--seed`. Realizations
left with fewer than `--min-particles` surviving particles are dropped and
reported separately. The window is recorded in `config.<i>.json` and as
`phi1_min`/`phi1_max`/`phi2_min`/`phi2_max` HDF5 attributes (`nan` where
unbounded).

## Train

```bash
python train_npe.py --config configs/chebconv_9params.py
```

Config fields worth knowing:

- `config.workdir` — shared root across every run of every project (e.g.
  `/scratch/$USER/trained_models/npe`), **not** project-specific.
  `config.wandb_project` is the per-project name; `WandbLogger` nests
  `workdir/<wandb_project>/<run_id>/checkpoints/` on its own, so folding
  the project name into `workdir` too causes double nesting.
- `config.checkpoint = 'last.ckpt'` + `config.id = '<fixed run id>'` —
  resume this exact run. Both must be set together: `config.id` fixed is
  what makes `project_dir` (and therefore `last.ckpt`'s location)
  deterministic across resubmissions. Leave both unset to start fresh.
- `config.reset_optimizer` — `False` (default) does a full resume
  (optimizer/scheduler/epoch/RNG state all continue); `True` loads weights
  only and starts training fresh from them (use for transfer learning, not
  routine resumes).

On resume, the checkpoint's own recorded `norm_dict` is always reused
(never recomputed from data) — see `train_npe.py`'s `main()`.

### On SLURM

```bash
./slurm/submit.sh configs/chebconv_9params.py
./slurm/submit.sh configs/chebconv_9params.py --time=1-00:00:00 --partition=compute_h200
./slurm/submit.sh configs/chebconv_9params.py --config.train_batch_size=128
```