
import numpy as np
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 468132
    config.seed_training = np.random.randint(0, 1_000_000)

    # data configuration
    config.data_root = '/scratch/tvnguyen/stream_datasets/'
    config.data_name = '9p_AAU'
    config.num_datasets = 20
    config.feat_labels = ('phi1', 'phi2', 'vr', 'pm1', 'pm2', 'dist')
    config.labels = (
        'log_mass', 'log_radius', 'v_rel_perp', 'v_rel_para', 'angle_pos_impact',
        'angle_vel_delta', 'impact_param', 'time_impact', 'phi1_impact_today'
    )
    config.train_frac = 0.9
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.wandb_project = '9p_AAU'
    config.workdir = '/scratch/tvnguyen/trained_models/npe'
    config.entity = "desc_sbi_stream"
    config.name = None
    config.id = None
    config.tags = ['npe', 'chebconv', 'nsf']
    config.checkpoint = None
    config.reset_optimizer = False
    config.debug = False
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 10
    model.output_size = len(config.labels)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.type = 'gnn'
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = {'K': 8}
    model.embedding.gnn.hidden_sizes = [128, ] * 5
    model.embedding.gnn.act_name = 'relu'
    model.embedding.gnn.pooling = "mean"
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [128, ]
    model.embedding.mlp.output_size = 128
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.dropout = 0.0

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.type = 'nsf'
    model.flows.num_transforms = 6
    model.flows.hidden_features = [128, 128]
    model.flows.activation = 'tanh'
    model.flows.num_bins = 8
    model.flows.randperm = True

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True if model.embedding.type == 'gnn' else False
    pre_transforms.apply_projection = False
    pre_transforms.apply_selection = False
    pre_transforms.apply_uncertainty = False
    pre_transforms.use_log_features = False
    pre_transforms.recompute_node_features = False
    pre_transforms.graph_name = 'adaptive_knn'
    pre_transforms.graph_args = {'ratio': 0.2, 'loop': True}

    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = True
    config.visualization = visualization = ConfigDict()
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 500
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = int(900_000 * 2 *  0.9 * 100 / 128)
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING configuration ###
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 128
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.patience = 100
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
