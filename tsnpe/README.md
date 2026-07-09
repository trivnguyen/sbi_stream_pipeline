# TSNPE pipeline

Truncated Sequential Neural Posterior Estimation (Deistler et al. 2022) on
top of `jgnn`. A run's model checkpoints and observational data are
tracked in a `state.json` manifest so later rounds can't use the wrong one.

## Config

`configs/aau.py` is the one config file, self-contained (no base/
override split). Edit it directly to point at a different target or
change any setting.

## Layout

```
configs/aau.py     the config
tsnpe/
  state.py       run-state manifest (state.json read/write; see below)
  target.py      load + hard-copy a target's observational data
  proposal.py    TSNPE truncated-proposal sampler (real observation in,
                 truncated proposal out; no conditioning term - this
                 model's embedding has none)
  model_io.py    rebuild an NPE model from a stored architecture config,
                 or a small fixed one for debug_model_config()
register_run.py     one-time: register target + round-0 model
simulate_round.py   round r >= 1: proposal sample + Agama/Nbody_streams simulate
train_round.py      round r >= 1: fine-tune round r-1 on round r's data
run_pipeline.sh      register once, then loop rounds
```

The prior box, simulator, and coordinate transforms live in the
repo-root [`stream_sims/`](../stream_sims/) package (shared with `npe/`),
not here — `tsnpe/tsnpe/` only has the pieces specific to the truncated,
target-conditioned rounds.

## Round semantics

- Round 0 is a pretrained wide-prior checkpoint, registered (not trained)
  by `register_run.py`.
- Round r >= 1 simulates a truncated proposal from round r-1's model, then
  fine-tunes round r-1's checkpoint on *only* round r's fresh simulations.
- The normalization dict and model architecture are fixed at round 0 and
  reused verbatim by every later round.

## state.json

Every script reads/writes `<run_dir>/state.json` instead of taking
checkpoint/x_obs paths as CLI flags. Paths are relative to `run_dir` and
point at hard copies, so a run directory is self-contained:

```json
{
  "seed": 0,
  "target": {"npz_path": "target/x_obs.npz", "sha256": "...", "key": "..."},
  "base": {
    "checkpoint_path": "round_0/model.ckpt",
    "norm_dict_path": "round_0/norm_dict.json",
    "model_config_path": "round_0/model_config.json",
    "source": "wandb", "wandb_run_path": "<entity>/<project>/<run_id>"
  },
  "rounds": {
    "1": {
      "data_path": "round_1/data.hdf5",
      "diagnostics": {"tau": -12.3, "acceptance_rate": 0.004},
      "checkpoint_path": "round_1/jgnn-tsnpe/<run_id>/checkpoints/last.ckpt",
      "wandb_run_id": "abc123"
    }
  }
}
```

Every script no-ops against an already-registered/already-run step, so
`run_pipeline.sh` just always calls every step and resumes correctly after
a partial failure.

## The prior is fixed, not configured

The prior box (9 physical params — see `stream_sims/prior.py`'s module
docstring) is unlikely to change, so it's plain constants there, not a
config file. There is no conditioning dimension in this model (unlike the
`sbi_dsph` sibling project's `stellar_log_r_star` conditioning) —
`tsnpe/proposal.py`'s embedding/log-prob calls reflect that directly, no
MC-conditioning marginalization.

## Debug mode

`config.pretrained.random_init = True` builds round 0 from a small fixed
architecture (`tsnpe.model_io.debug_model_config`) and a norm_dict that
doesn't depend on any real data (`stream_sims.prior.default_norm_dict`) —
no wandb, no target needed first. Posteriors/proposals from this are
meaningless; it only exercises the pipeline's plumbing.

## Usage

```bash
# Step by step:
python register_run.py    --config configs/aau.py
python simulate_round.py  --config configs/aau.py --config.round=1
python train_round.py     --config configs/aau.py --config.round=1

# Or all at once, rounds 1..5:
./run_pipeline.sh --config configs/aau.py --rounds 5

# Any ml_collections override is passed through, e.g. more sims per round:
./run_pipeline.sh --config configs/aau.py --rounds 5 \
    --config.proposal.n_sims=2000
```
