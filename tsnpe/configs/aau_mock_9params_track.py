""" AAU subhalo config: real round-0 model, checkpoint + config from wandb.

config.pretrained.local_checkpoint_dir controls only where the checkpoint
*file* comes from: leave it empty to download it from wandb, or point it
at an already-downloaded copy (e.g. a cached
'artifacts/model-<run_id>:v<N>' directory) to skip the download. Either
way, the architecture and pre_transforms recipe are always read from
wandb (config.pretrained.wandb_run_path) and cached locally at
registration (round_0/model_config.json, round_0/pre_transforms_config.json)
- never hand-copied here, since that risks drifting from what actually
trained the checkpoint. Set config.pretrained.pre_transforms_override to
use a different recipe instead (e.g. a checkpoint trained before
pre_transforms was logged to wandb).

    python register_run.py    --config configs/draco.py
    python simulate_round.py  --config configs/draco.py --config.round=1
    python train_round.py     --config configs/draco.py --config.round=1
"""

import numpy as np
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.run_dir = '/scratch/tvnguyen/trained_models/tsnpe/aau_mock_9params_track'
    config.seed = 42
    config.round = placeholder(int)
    config.overwrite = False

    config.target = ConfigDict()
    config.target.catalog_path = '/scratch/tvnguyen/stream_datasets/observations/aau_mock_v1.csv'
    config.target.key = 'aau_mock_v1'
    config.target.true_params = ConfigDict(dict(
        log_mass=0.0,
        log_radius=np.log10(0.3),
        v_rel_perp=35.0,
        v_rel_para=10.0,
        angle_pos_impact=0.0,
        angle_vel_delta=45.0,
        impact_param=1.0,
        time_impact=0.25,
        phi1_impact_today=-8.0,
    ))

    config.pretrained = ConfigDict()
    config.pretrained.random_init = False  # for debug only
    config.pretrained.wandb_run_path = 'desc_sbi_stream/jgnn-tsnpe/runs/dl04wm57'
    config.pretrained.wandb_version = 'best'
    config.pretrained.local_checkpoint_dir = '/scratch/tvnguyen/trained_models/npe/9p_AAU_deltaVmin0p03/dl04wm57/checkpoints'
    config.pretrained.local_checkpoint_filename = 'last.ckpt'

    config.proposal = ConfigDict()
    config.proposal.n_sims = 2000
    config.proposal.epsilon = 1e-3
    config.proposal.n_post_samples = 100_000
    config.proposal.draw_batch = 20_000
    config.proposal.oversample_cap = 10000
    config.proposal.delta_vmin = 0.03
    config.proposal.sampling_mode = 'sir'

    config.simulation = ConfigDict()
    config.simulation.n_particles = 250
    config.simulation.n_jobs = 0
    config.simulation.use_multiprocessing = True
    config.simulation.sample_threads = 1

    config.training = ConfigDict()
    # Set dynamically per round by train_round.py (round_dir, not a fixed
    # shared path - each round's checkpoints/wandb files stay
    # self-contained under run_dir/round_<r>/). id is likewise set
    # dynamically when resuming a previously-recorded run.
    config.training.workdir = placeholder(str)
    config.training.id = placeholder(str)
    config.training.wandb_project = 'jgnn-tsnpe'
    config.training.entity = 'desc_sbi_stream'
    config.training.debug = False
    config.training.enable_progress_bar = True
    config.training.accelerator = 'gpu'
    config.training.train_batch_size = 128
    config.training.eval_batch_size = 128
    config.training.train_frac = 0.9
    config.training.num_workers = 0
    config.training.num_epochs = 40
    config.training.num_steps = -1
    config.training.patience = 20
    config.training.gradient_clip_val = 0.5

    config.training.optimizer = ConfigDict()
    config.training.optimizer.name = 'AdamW'
    config.training.optimizer.lr = 1e-4
    config.training.optimizer.weight_decay = 0.01

    config.training.scheduler = ConfigDict()
    config.training.scheduler.name = 'WarmUpCosineAnnealingLR'
    config.training.scheduler.decay_steps = 40
    config.training.scheduler.warmup_steps = 2
    config.training.scheduler.eta_min = 1e-2
    config.training.scheduler.interval = 'epoch'

    config.training.enable_visualization_callback = True
    config.training.visualization = ConfigDict()
    config.training.visualization.n_posterior_samples = 500
    config.training.visualization.n_val_samples = 1000
    config.training.visualization.plot_every_n_epochs = 1
    config.training.visualization.plot_tarp = True
    config.training.visualization.plot_median_v_true = True
    config.training.visualization.plot_rank = True

    return config
