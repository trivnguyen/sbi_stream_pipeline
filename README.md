# sbi_stream_pipeline

Simulation-based inference pipeline for stellar-stream perturber (subhalo
impact) parameters, using graph neural network posterior estimation
(`jgnn`). Three pieces, kept in one repo because they're tightly coupled
through shared simulation code and checkpoint/config formats:

1. **[`stream_sims/`](stream_sims/)** — the shared simulator: prior box,
   Agama + `Nbody_streams` perturbed-stream physics, and HDF5 graph-dataset
   I/O. Used by both stages below; owns the one canonical definition of the
   9 free parameters and the simulation model.
2. **[`npe/`](npe/README.md)** — train a baseline amortized Neural
   Posterior Estimator on wide-prior training data.
3. **[`tsnpe/`](tsnpe/README.md)** — Truncated Sequential NPE (Deistler et
   al. 2022): starting from `npe`'s checkpoint, iteratively simulate a
   truncated proposal around a real target observation and fine-tune,
   narrowing the posterior over several rounds.

See each subdirectory's own README for full usage.

## Dependencies

- `jgnn` — GNN embeddings, the NPE model, shared PyTorch Lightning
  training utilities. A separate repo/package, not vendored here. Install
  it editable (`pip install -e /path/to/jgnn`) before running anything in
  either `npe/` or `tsnpe/`.
- `agama` — galactic-dynamics potentials and orbit integration.
- `Nbody_streams` — particle-spray stream simulation (external local
  package; needs `numba`).

All of the above live in the `torch-jax` venv on this cluster, not
`torch` (which only has `jgnn`, for the `sbi_dsph` sibling project).

## Layout

```
stream_sims/   shared simulator - prior.py (9-param prior box + detectability
               cut), sims.py (Agama + Nbody_streams simulate_one /
               run_simulation_batch / write_graph_dataset), sims_utils.py
               (coordinate transforms), potentials/ (fixed MW+LMC potential,
               unperturbed stream snapshot, stripping-time distribution).
npe/           baseline NPE trainer - simulate wide-prior training data, train
               the amortized model, submit to SLURM.
tsnpe/         truncated sequential NPE - register npe's checkpoint, then round
               by round: simulate a truncated proposal, fine-tune.
```

`npe` and `tsnpe` both import `stream_sims` (via a repo-root
`sys.path.insert`, since this repo has no `pyproject.toml`/`setup.py`) —
there is exactly one copy of the prior and simulator, not two drifting
copies.

## Workflow

```bash
# 1. Simulate npe's training data (wide prior, no target observation
#    involved yet)
cd npe
python simulate_process.py --n-sims 100000 --n-workers 24 \
    --output-dir /scratch/$USER/stream_datasets/9p_AAU

# 2. Train the baseline NPE - locally, or via slurm/submit.sh on a cluster
python train_npe.py --config configs/chebconv_9params.py

# 3. Feed that checkpoint into tsnpe's truncated rounds against a real
#    target observation (config.pretrained points at npe's wandb run or
#    a local checkpoint - see tsnpe/README.md)
cd ../tsnpe
python register_run.py --config configs/aau.py
./run_pipeline.sh --config configs/aau.py --rounds 5
```

`npe` and `tsnpe` are otherwise independent to run — `tsnpe` only ever
*reads* a finished `npe` checkpoint (plus its recorded `norm_dict` and
pre-transforms config), it never modifies `npe`'s output.
