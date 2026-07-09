"""TSNPE (Deistler et al. 2022) truncated-proposal sampler.

Reads the real target's observation, and computes the truncated proposal:
1. embed the observation once (see _embed_observation) - this model has
   no conditioning input, so model.forward(batch) is already the full,
   theta-independent embedding
2. sample the posterior, estimate tau from it (estimate_tau)
3. sample the prior and cut at tau, or sampling-importance-resample
   directly from the posterior (sample_tsnpe_proposal)
"""

import numpy as np
import torch
from scipy.special import logsumexp
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from jgnn.transforms import build_transformation

from stream_sims import prior as prior_lib
from .target import TargetData

_PRIOR = prior_lib.Prior()


def build_obs_pre_transforms(pre_transforms_config: dict, norm_dict: dict):
    """Pre-transforms for a real observation graph.

    Graph construction and normalization exactly match the model's own
    training config (`pre_transforms_config`, e.g. config.pre_transforms) -
    only the augmentations that assume raw pos/vel or a training-style
    batch (projection, selection, uncertainty, node-feature recompute) are
    forced off, since the observation's x is already computed manually
    (_x_obs_features) and it's a single graph, not a batch.
    """
    return build_transformation(norm_dict=norm_dict, **{
        **pre_transforms_config,
        'apply_projection': False,
        'apply_selection': False,
        'apply_uncertainty': False,
        'recompute_node_features': False,
    })


def _x_obs_features(target: TargetData) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (x, pos) node-feature tensors for the observed stream.

    Blocked on tsnpe.target.TargetData's stream-specific redesign (not
    implemented yet). Likely shape once that lands: pos=(phi1, phi2)
    (spatial), x=(dist, pm1, pm2, vr[, uncertainties]) (mirroring
    NODE_FEATURE_NAMES in sims.py, minus the spatial columns).
    """
    raise NotImplementedError(
        "_x_obs_features depends on tsnpe.target.TargetData's stream "
        "redesign, which is not implemented yet.")


def _embed_observation(model, obs_graph) -> torch.Tensor:
    """Compute the observation's embedding exactly once.

    Reused for the entire tau-calibration + proposal-sampling call below -
    there is no per-candidate conditioning term in this project's model
    (see module docstring), so the embedding is fully theta-independent
    and this is the model's plain public forward pass, no internals-
    reaching-into required.
    """
    obs_batch = Batch.from_data_list([obs_graph])
    with torch.no_grad():
        return model.forward(obs_batch)  # (1, embedding_output_size)


def _log_prob_candidates(
    model, graph_embedding: torch.Tensor, norm_dict: dict, theta_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate log q(theta | x_obs) for a batch of physical-unit theta,
    given the already-computed observation embedding.
    """
    theta_loc = np.asarray(norm_dict['theta_loc'])
    theta_scale = np.asarray(norm_dict['theta_scale'])
    theta_norm = (theta_batch - theta_loc) / theta_scale
    with torch.no_grad():
        theta_tensor = torch.tensor(
            theta_norm, dtype=torch.float32, device=graph_embedding.device)
        embedding_batch = graph_embedding.expand(len(theta_tensor), -1)
        log_prob = model.flows(embedding_batch).log_prob(theta_tensor)
    return log_prob.cpu().numpy()


def sample_posterior(
    model,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    n_samples: int = 2000,
    return_log_prob: bool = False,
) -> np.ndarray:
    """Draw posterior samples at the real target's observation, for diagnostics.

    No conditioning dimension to marginalize over: draws come directly
    from the single model.flows(graph_embedding) distribution.

    Args:
        model: Trained NPE model (jgnn.models.NPE).
        target: Observational data snapshot.
        norm_dict: Normalization dict matching the model.
        pre_transforms_config: The model's own pre_transforms config (e.g.
            config.pre_transforms) - see build_obs_pre_transforms.
        n_samples: Number of posterior draws.
        return_log_prob: If True, also return the log-density of each
            posterior sample under the model.

    Returns:
        (n_kept, 9) ndarray, physical units, columns matching
        tsnpe.prior.PARAM_NAMES (n_kept <= n_samples: samples outside
        the normalized prior box are dropped).

    Raises:
        RuntimeError: If every posterior sample falls outside the prior box.
    """
    model.eval()
    pre_transforms = build_obs_pre_transforms(pre_transforms_config, norm_dict)
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))
    graph_embedding = _embed_observation(model, obs_graph)

    with torch.no_grad():
        dist = model.flows(graph_embedding)
        if return_log_prob:
            post, log_q = dist.rsample_and_log_prob((n_samples,))
            log_q = log_q.reshape(-1).cpu().numpy()
        else:
            post = dist.rsample((n_samples,))
    post_norm = post.reshape(-1, post.shape[-1]).cpu().numpy()

    # normalized prior is always [-1, 1] in every dimension, so the cut is simple.
    in_box = np.all((post_norm >= -1) & (post_norm <= 1), axis=1)
    if not in_box.any():
        raise RuntimeError(
            'All posterior samples fell outside the prior box. The '
            'previous round\'s model may not have converged.')
    post_norm = post_norm[in_box]

    theta_loc = np.asarray(norm_dict['theta_loc'])
    theta_scale = np.asarray(norm_dict['theta_scale'])
    post_phys = post_norm * theta_scale + theta_loc

    if return_log_prob:
        return post_phys, log_q[in_box]
    return post_phys


def estimate_tau(
    model,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    epsilon: float = 1e-3,
    n_post_samples: int = 50_000,
    return_posterior: bool = False,
):
    """Calibrate tau, the epsilon-quantile of in-box posterior log-density.

    Tau is a property of the posterior, so it's estimated from exactly the
    same posterior draws sample_posterior would produce - the only extra
    step here is taking the epsilon-quantile of their log-density.
    `return_posterior` reuses those same draws instead of requiring a
    second, redundant sample_posterior call for diagnostics.

    Args:
        return_posterior: If True, also return the posterior samples
            already drawn for calibration.

    Returns:
        tau, or (tau, posterior_phys) if return_posterior - posterior_phys
        is an (n_kept, 9) ndarray, physical units, columns matching
        tsnpe.prior.PARAM_NAMES.

    Raises:
        RuntimeError: If every posterior sample falls outside the prior box.
    """
    posterior, log_q = sample_posterior(
        model, target, norm_dict, pre_transforms_config,
        n_samples=n_post_samples, return_log_prob=True)

    tau = float(np.quantile(log_q, epsilon))
    print(f'  tau (eps={epsilon:.0e}): {tau:.4f} '
          f'[log-prob: {log_q.min():.2f} .. {log_q.max():.2f}]')

    if return_posterior:
        return tau, posterior
    return tau


def _sample_proposal_rejection(
    model, graph_embedding: torch.Tensor, norm_dict: dict, tau: float,
    n_sims: int, draw_batch: int, oversample_cap: int,
):
    """Rejection-sample the prior, keeping candidates whose log-density
    under the model is >= tau (Deistler et al. 2022). Because the proposal
    is the prior truncated to this set, it is a proper distribution and
    standard NLL training needs no importance correction.

    Candidates come straight from tsnpe.prior.Prior.sample_prior - already
    physical-unit, already delta_V > 3 km/s filtered.

    Returns:
        (proposal_phys, diagnostics) - proposal_phys: (n_sims, 9) ndarray,
        physical units, columns matching tsnpe.prior.PARAM_NAMES;
        diagnostics: dict with acceptance_rate, n_drawn, n_accepted.

    Raises:
        RuntimeError: If no prior candidates pass the tau filter.
    """
    n_max = n_sims * oversample_cap
    accepted = []
    n_accepted, n_drawn = 0, 0

    pbar = tqdm(total=n_sims, desc='Sampling proposal', unit='accepted')
    while n_accepted < n_sims and n_drawn < n_max:
        cands_phys = _PRIOR.sample_prior(draw_batch)
        lq = _log_prob_candidates(model, graph_embedding, norm_dict, cands_phys)
        mask = lq >= tau
        if mask.any():
            accepted.append(cands_phys[mask])
            n_accepted += int(mask.sum())
        n_drawn += draw_batch
        pbar.update(int(mask.sum()))
    pbar.close()

    if n_accepted == 0:
        raise RuntimeError(
            f'No candidates passed tau={tau:.3f} after {n_drawn:,} draws. '
            'Try raising epsilon or increasing oversample_cap.')

    acc_rate = n_accepted / n_drawn
    print(f'  Prior acceptance: {acc_rate:.4e} '
          f'(drawn={n_drawn:,}, accepted={n_accepted:,})')

    proposal_phys = np.concatenate(accepted)[:n_sims]
    diagnostics = dict(
        acceptance_rate=float(acc_rate),
        n_drawn=int(n_drawn), n_accepted=int(n_accepted))
    return proposal_phys, diagnostics


def _sample_proposal_sir(
    model, graph_embedding: torch.Tensor, norm_dict: dict, tau: float,
    n_sims: int, draw_batch: int, oversample_cap: int,
):
    """Sampling-importance-resampling directly from the posterior.

    There's no per-draw conditioning value to marginalize over, so draws
    come repeatedly from the single model.flows(graph_embedding)
    distribution, weighted by
    1{log_q >= tau} / q (uniform prior, so weights need no extra
    prior-density factor) and resampled.

    Can be far more sample-efficient than _sample_proposal_rejection when
    the tau-truncated region is small relative to the prior box (low prior
    acceptance rate), at the cost of the resampled draws being only
    approximately i.i.d. from the truncated prior (importance-weight
    degeneracy, tracked via ess_total below).

    Returns:
        (proposal_phys, diagnostics) - proposal_phys: (n_sims, 9) ndarray,
        physical units, columns matching tsnpe.prior.PARAM_NAMES;
        diagnostics: dict with ess_total (effective sample size of the
        accumulated weighted draws) and n_drawn/n_total.
    """
    theta_loc = np.asarray(norm_dict['theta_loc'])
    theta_scale = np.asarray(norm_dict['theta_scale'])

    ess_total = 0.0
    n_drawn = 0
    theta_running, logw_running = [], []

    pbar = tqdm(total=n_sims, desc='Sampling proposal (SIR)', unit='neff')
    while ess_total < n_sims and n_drawn < n_sims * oversample_cap:
        with torch.no_grad():
            post, log_q = model.flows(graph_embedding).rsample_and_log_prob((draw_batch,))
        post_norm = post.reshape(-1, post.shape[-1]).cpu().numpy()
        log_q = log_q.reshape(-1).cpu().numpy()
        theta = post_norm * theta_scale + theta_loc

        in_tau = log_q >= tau
        logw = np.where(in_tau, -log_q, -np.inf)  # w propto 1{in S} / q (uniform prior)
        theta_running.append(theta)
        logw_running.append(logw)
        n_drawn += draw_batch

        logw_cat = np.concatenate(logw_running)
        w = np.exp(logw_cat - logsumexp(logw_cat))
        ess_total = 1.0 / np.sum(w ** 2)
        pbar.update(int(ess_total) - pbar.n)
    pbar.close()

    theta_running = np.concatenate(theta_running)
    logw_running = np.concatenate(logw_running)
    w = np.exp(logw_running - logsumexp(logw_running))

    idx = np.random.choice(len(theta_running), size=n_sims, replace=True, p=w)
    proposal_phys = theta_running[idx]

    print(f'  SIR effective sample size: {ess_total:.1f} '
          f'(drawn={n_drawn:,}, total={len(theta_running):,})')

    diagnostics = dict(
        ess_total=float(ess_total), n_drawn=int(n_drawn),
        n_total=int(len(theta_running)))
    return proposal_phys, diagnostics


def sample_tsnpe_proposal(
    model,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    n_sims: int = 1000,
    epsilon: float = 1e-3,
    n_post_samples: int = 50_000,
    draw_batch: int = 10_000,
    oversample_cap: int = 500,
    sampling_mode: str = 'rejection',
    return_posterior: bool = False,
):
    """Draw a TSNPE-truncated proposal for the next simulation round.

    Estimates tau (estimate_tau), then draws n_sims samples from the
    tau-truncated region via sampling_mode:
    - 'rejection' (default): rejection-sample the prior - see
      _sample_proposal_rejection. Yields a proper distribution, so
      standard NLL training needs no importance correction.
    - 'sir': sampling-importance-resampling directly from the posterior -
      see _sample_proposal_sir. Can be far more sample-efficient when the
      prior acceptance rate is low, at the cost of only approximate i.i.d.
      draws (importance-weight degeneracy).

    Args:
        model: Trained NPE model (jgnn.models.NPE).
        target: Observational data snapshot.
        norm_dict: Fixed normalization dict (the same one used at round 0,
            never recomputed across rounds).
        pre_transforms_config: The model's own pre_transforms config (e.g.
            config.pre_transforms) - see build_obs_pre_transforms.
        n_sims: Number of proposal draws to return.
        epsilon: Posterior mass fraction excluded when calibrating tau.
        n_post_samples: Total posterior samples drawn to calibrate tau.
        draw_batch: Candidates/posterior draws per sampling_mode batch.
        oversample_cap: Hard ceiling on draws = n_sims * oversample_cap.
        sampling_mode: 'rejection' or 'sir' - see above.
        return_posterior: If True, also return the posterior samples drawn
            for tau calibration (see estimate_tau) - avoids a second,
            redundant sample_posterior call for diagnostics.

    Returns:
        (proposal_phys, diagnostics), or (proposal_phys, diagnostics,
        posterior_phys) if return_posterior:
        - proposal_phys: (n_sims, 9) ndarray, physical units, columns
          matching tsnpe.prior.PARAM_NAMES.
        - diagnostics: dict with tau, n_drawn, plus sampling_mode-specific
          fields (acceptance_rate/n_accepted for 'rejection',
          ess_total/n_total for 'sir').
        - posterior_phys: (n_post_samples, 9) ndarray, same columns.

    Raises:
        RuntimeError: If every posterior sample falls outside the prior box,
            or (sampling_mode='rejection') no prior candidates pass the
            tau filter.
        ValueError: If sampling_mode isn't 'rejection' or 'sir'.
    """
    model.eval()
    tau_result = estimate_tau(
        model, target, norm_dict, pre_transforms_config,
        epsilon=epsilon, n_post_samples=n_post_samples,
        return_posterior=return_posterior)
    tau, posterior_phys = tau_result if return_posterior else (tau_result, None)

    pre_transforms = build_obs_pre_transforms(pre_transforms_config, norm_dict)
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))
    graph_embedding = _embed_observation(model, obs_graph)

    if sampling_mode == 'rejection':
        proposal_phys, diagnostics = _sample_proposal_rejection(
            model, graph_embedding, norm_dict, tau,
            n_sims=n_sims, draw_batch=draw_batch, oversample_cap=oversample_cap)
    elif sampling_mode == 'sir':
        proposal_phys, diagnostics = _sample_proposal_sir(
            model, graph_embedding, norm_dict, tau,
            n_sims=n_sims, draw_batch=draw_batch, oversample_cap=oversample_cap)
    else:
        raise ValueError(
            f"sampling_mode={sampling_mode!r} not recognized; "
            "must be 'rejection' or 'sir'.")

    diagnostics['tau'] = tau
    if return_posterior:
        return proposal_phys, diagnostics, posterior_phys
    return proposal_phys, diagnostics
