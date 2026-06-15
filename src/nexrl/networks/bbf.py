"""BBF network: scaled IMPALA encoder + dueling C51 head + SPR components.

"Bigger, Better, Faster: Human-level Atari with human-level efficiency"
(Schwarzer et al., ICML 2023)

Structure matches Schwarzer's reference SPRNetwork / RainbowDQNNetwork:

- The IMPALA encoder outputs a *spatial* latent ``(B, H', W', C)`` (no internal
  projection). ``renormalize`` (per-sample min-max to [0, 1]) is applied to it.
- A single network-level projection (``Dense``, NO activation) maps the
  flattened spatial latent to ``features_dim``. This raw projection is SHARED
  between the Q-head input and the SPR target/online representation. The ReLU
  is applied ONLY on the Q-head path (reference ``__call__`` does
  ``x = project(...); x = relu(x); head(x)`` — spr_networks.py:850-851), while
  the SPR target (``encode_project``) and SPR predictor input (``spr_predict``)
  consume the RAW projection (spr_networks.py:799-811). Baking ReLU into the
  shared projection would run the SPR cosine loss on non-negative vectors — a
  different auxiliary objective than the reference's.
- Dueling C51 head reads the projected features.
- SPR transition model is CONVOLUTIONAL (``ConvTMCell``): the action one-hot is
  broadcast across the spatial grid and concatenated on the channel axis, then
  two 3x3 convs + ReLU, followed by ``renormalize``. The rollout transitions in
  spatial latent space and projects each predicted latent before the predictor.
- SPR predictor is a single Linear ``F -> F``.

When ``encoder`` is None, expects flat observations and uses a linear embedding
for ``encode`` (useful for non-image Q-only sanity checks). The convolutional
SPR path requires a spatial encoder.

Dimension key:
    B: batch size
    C: input channels (frame_stack)
    H: image height
    W: image width
    Hs, Ws: spatial latent height/width (after encoder downsampling)
    Cs: spatial latent channels (= 32 * width_scale)
    A: action dimension (num_actions)
    N: number of atoms (n_atoms, C51)
    F: features_dim (projection output = SPR projection dim)
    J: jumps (SPR prediction horizon)
"""

from typing import Any

import distrax
import jax
from flax import nnx
from jax import numpy as jnp


def renormalize(x: jax.Array) -> jax.Array:
    """Schwarzer's ``renormalize``: per-sample min-max normalize to [0, 1].

    Normalizes over all non-batch dims jointly. Computed in fp32 for numerical
    stability (min-max + division), returned in the input dtype so the bf16
    compute path is preserved.

    Args:
        x: Spatial latent ``(B, Hs, Ws, Cs)`` (or any ``(B, ...)`` tensor).

    Returns:
        Min-max normalized tensor, same shape and dtype as ``x``.
    """
    in_dtype = x.dtype
    flat = x.astype(jnp.float32).reshape(x.shape[0], -1)
    min_B1 = flat.min(axis=-1, keepdims=True)
    max_B1 = flat.max(axis=-1, keepdims=True)
    normed = (flat - min_B1) / (max_B1 - min_B1 + 1e-5)
    return normed.reshape(x.shape).astype(in_dtype)


def _gather_action_atoms(x_BAN: jax.Array, action_B: jax.Array) -> jax.Array:
    """Gather the chosen action's atoms from a per-action distribution.

    Single source of truth for the C51 action-atom gather. Used by
    ``BBFNetwork.take_probs`` / ``take_logprobs`` (action selection / log-prob
    extraction) and by ``bbf.train_bbf``'s CE loss to pull the chosen action's
    log-probs from already-computed ``log_probs_BAN`` without recomputing logits.

    Args:
        x_BAN: Per-action atoms (probs or log-probs). ``(B, A, N)``.
        action_B: Chosen discrete actions. ``(B,)``.

    Returns:
        ``(B, N)`` — atoms of the selected action.
    """
    idx = action_B[:, None, None]
    return jnp.take_along_axis(x_BAN, jnp.broadcast_to(idx, (*action_B.shape, 1, x_BAN.shape[-1])), axis=1).squeeze(1)


class BBFNetwork(nnx.Module):
    """BBF agent network: scaled IMPALA encoder + dueling C51 + convolutional SPR.

    The network-level projection (``Dense`` on the flattened, renormalized
    spatial latent — NO activation) is shared between the Q-head input and the
    SPR representation, matching Schwarzer's reference where the ``FeatureLayer``
    projection feeds both the Q-head and the SPR projection/target. The ReLU is
    applied only on the Q-head path (see ``encode`` / the loss_fn Q-head site),
    not inside ``project`` — so the SPR target/predictor see the raw projection.
    """

    def __init__(
        self,
        action_dim: int,
        n_atoms: int,
        v_min: float,
        v_max: float,
        features_dim: int,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
        in_features: int = 0,
        *,
        compute_dtype: jnp.dtype = jnp.float32,
        kernel_init: nnx.Initializer | None = None,
    ):
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self._has_encoder = encoder is not None
        self._compute_dtype = compute_dtype

        # Linear/Conv compute in compute_dtype; params (master weights) stay fp32.
        # kernel_init=None -> NNX default (lecun_normal); BBF passes xavier_uniform
        # to match Schwarzer's reference (every conv/dense is xavier_uniform). This
        # init also flows to the S&P reset's fresh draws (make_fresh_params builds a
        # BBFNetwork with the same kwargs), so resets re-draw from the matched init.
        lin_kwargs: dict[str, Any] = {"dtype": compute_dtype, "param_dtype": jnp.float32}
        if kernel_init is not None:
            lin_kwargs["kernel_init"] = kernel_init

        # --- Encoder (spatial) + shared projection ---
        if encoder is not None:
            self.encoder = encoder
            spatial_h, spatial_w, spatial_c = encoder.spatial_shape  # pyright: ignore[reportAttributeAccessIssue]
            self._spatial_channels = spatial_c
            spatial_dim = spatial_h * spatial_w * spatial_c

            # Shared projection: flatten(renormalized spatial) -> features_dim.
            self.projection = nnx.Linear(spatial_dim, features_dim, rngs=rngs, **lin_kwargs)

            # --- SPR transition model: ConvTMCell on spatial latents ---
            # Action one-hot is broadcast over the spatial grid and concatenated
            # on the channel axis; two 3x3 SAME convs back to Cs channels.
            self.transition_conv1 = nnx.Conv(
                spatial_c + action_dim, spatial_c, kernel_size=(3, 3), padding="SAME", rngs=rngs, **lin_kwargs
            )
            self.transition_conv2 = nnx.Conv(spatial_c, spatial_c, kernel_size=(3, 3), padding="SAME", rngs=rngs, **lin_kwargs)
        else:
            self.flat_embed = nnx.Linear(in_features, features_dim, rngs=rngs, **lin_kwargs)

        # --- Dueling C51 head: reads projected features ---
        self.value_head = nnx.Linear(features_dim, n_atoms, rngs=rngs, **lin_kwargs)
        self.advantage_head = nnx.Linear(features_dim, action_dim * n_atoms, rngs=rngs, **lin_kwargs)

        # --- SPR predictor: single Linear (matches reference nn.Dense) ---
        self.predictor = nnx.Linear(features_dim, features_dim, rngs=rngs, **lin_kwargs)

    @property
    def support(self) -> jax.Array:
        return jnp.linspace(self.v_min, self.v_max, self.n_atoms)

    def encode_spatial(self, obs_BCHW: jax.Array) -> jax.Array:
        """Encode to renormalized spatial latent. (B, C, H, W) -> (B, Hs, Ws, Cs)."""
        assert self._has_encoder, "encode_spatial requires a spatial CNN encoder"
        return renormalize(self.encoder(obs_BCHW))

    def project(self, spatial_BHWC: jax.Array) -> jax.Array:
        """Shared RAW projection: flatten spatial latent -> Dense (no activation). (B, Hs, Ws, Cs) -> (B, F).

        Reference ``project`` (spr_networks.py:804-806) applies no activation; the
        ReLU lives only on the Q-head path (``encode`` / loss_fn). SPR target
        (``encode_project``) and predictor input (``spr_predict``) consume this raw
        projection, so the SPR cosine loss runs on the same space as the reference.
        """
        flat_BS = spatial_BHWC.reshape(spatial_BHWC.shape[0], -1)
        return self.projection(flat_BS)

    def encode(self, obs: jax.Array) -> jax.Array:
        """Encode observations to Q-head feature vectors. (B, C, H, W) or (B, D) -> (B, F).

        CNN path: encoder spatial latent -> renormalize -> flatten -> projection -> ReLU.
        The ReLU is applied HERE (Q-head path), not inside ``project`` — matching the
        reference ``__call__`` (``project`` then ``relu`` before the head). The raw
        ``project`` output feeds the SPR target/predictor instead (see ``project``).
        """
        if self._has_encoder:
            return nnx.relu(self.project(self.encode_spatial(obs)))
        return nnx.relu(self.flat_embed(obs.reshape(obs.shape[0], -1).astype(jnp.float32)))

    def q_logits(self, features_BF: jax.Array) -> jax.Array:
        """Dueling C51 logits from projected features. (B, F) -> (B, A, N)."""
        value_BN = self.value_head(features_BF)  # (B, N)
        advantage_BAN = self.advantage_head(features_BF).reshape(-1, self.action_dim, self.n_atoms)
        advantage_BAN = advantage_BAN - jnp.mean(advantage_BAN, axis=1, keepdims=True)
        return value_BN[:, None, :] + advantage_BAN  # (B, A, N)

    def q_values(self, obs_BCHW: jax.Array) -> jax.Array:
        """Expected Q-values: encode -> q_logits -> softmax -> E[support]. (B, C, H, W) -> (B, A)."""
        features_BF = self.encode(obs_BCHW)
        # softmax/log_softmax must be fp32 — bf16 collapses categorical CE (Flax #677).
        logits_BAN = self.q_logits(features_BF).astype(jnp.float32)
        probs_BAN = jax.nn.softmax(logits_BAN, axis=-1)
        return jnp.sum(probs_BAN * self.support, axis=-1)  # (B, A)

    def probs(self, features_BF: jax.Array) -> jax.Array:
        """Action-atom probabilities from features. (B, F) -> (B, A, N)."""
        return jax.nn.softmax(self.q_logits(features_BF).astype(jnp.float32), axis=-1)

    def take_probs(self, features_BF: jax.Array, action_B: jax.Array) -> jax.Array:
        """Probs for selected actions. (B, F), (B,) -> (B, N)."""
        return _gather_action_atoms(self.probs(features_BF), action_B)

    def take_logprobs(self, features_BF: jax.Array, action_B: jax.Array) -> jax.Array:
        """Log-probs for selected actions. (B, F), (B,) -> (B, N)."""
        logits_BAN = self.q_logits(features_BF).astype(jnp.float32)
        log_probs_BAN = jax.nn.log_softmax(logits_BAN, axis=-1)
        return _gather_action_atoms(log_probs_BAN, action_B)

    def transition(self, spatial_BHWC: jax.Array, action_B: jax.Array) -> jax.Array:
        """Convolutional transition (ConvTMCell): predict next spatial latent.

        Action one-hot is broadcast across the spatial grid and concatenated on
        the channel axis; two 3x3 convs + ReLU, then renormalize.

        (B, Hs, Ws, Cs), (B,) -> (B, Hs, Ws, Cs).
        """
        action_onehot_BA = jax.nn.one_hot(action_B, self.action_dim, dtype=self._compute_dtype)  # (B, A)
        # Broadcast onehot over the spatial grid: (B, A) -> (B, Hs, Ws, A)
        action_grid_BHWA = jnp.broadcast_to(action_onehot_BA[:, None, None, :], (*spatial_BHWC.shape[:-1], self.action_dim))
        x = jnp.concatenate([spatial_BHWC, action_grid_BHWA], axis=-1)  # (B, Hs, Ws, Cs+A)
        x = nnx.relu(self.transition_conv1(x))
        x = nnx.relu(self.transition_conv2(x))
        return renormalize(x)  # (B, Hs, Ws, Cs)

    def predict(self, features_BF: jax.Array) -> jax.Array:
        """SPR predictor: single Linear from features to features. (B, F) -> (B, F)."""
        return self.predictor(features_BF)

    def act(self, obs_BCHW: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        """Epsilon-greedy action selection. (B, C, H, W) -> (B,)."""
        q_BA = self.q_values(obs_BCHW)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


def spr_rollout(network: BBFNetwork, spatial_BHWC: jax.Array, actions_JB: jax.Array) -> jax.Array:
    """Roll out the convolutional transition model for J steps, returning predicted projections.

    Per step: spatial latent → conv transition(spatial, action) → project → predictor.
    The scan carry is the spatial latent (stays in compute_dtype since the conv
    transition + renormalize preserve dtype — no fp32-carry cast needed).

    Args:
        network: BBFNetwork (read-only — used inside jax.lax.scan).
        spatial_BHWC: Initial renormalized spatial latent. (B, Hs, Ws, Cs)
        actions_JB: Actions for each jump step. (J, B) int32.

    Returns:
        predictions_JBF: Predicted projected features per jump. (J, B, F)
    """

    def step(latent_BHWC: jax.Array, action_B: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_latent_BHWC = network.transition(latent_BHWC, action_B)
        predicted_BF = network.predict(network.project(next_latent_BHWC))
        return next_latent_BHWC, predicted_BF

    _, predictions_JBF = jax.lax.scan(step, spatial_BHWC, actions_JB)
    return predictions_JBF


def spr_targets(target_network: BBFNetwork, future_obs_JBCHW: jax.Array) -> jax.Array:
    """Compute SPR targets from target network (stop_gradient).

    Reference-aligned: SPR target is the RAW projection of the renormalized
    spatial latent (``encode_project``, no ReLU — spr_networks.py:799-802),
    matching the online predictor's raw-projection input (``spr_predict``).

    Args:
        target_network: Target BBFNetwork (read-only).
        future_obs_JBCHW: Future observations. (J, B, C, H, W)

    Returns:
        targets_JBF: Projected target features. (J, B, F)
    """
    J, B = future_obs_JBCHW.shape[:2]
    flat_obs = future_obs_JBCHW.reshape(J * B, *future_obs_JBCHW.shape[2:])
    spatial_BHWC = target_network.encode_spatial(flat_obs)
    projected_BF = target_network.project(spatial_BHWC)
    targets_JBF = projected_BF.reshape(J, B, -1)
    return jax.lax.stop_gradient(targets_JBF)


def spr_loss(
    predictions_JBF: jax.Array,
    targets_JBF: jax.Array,
    same_traj_mask_JB: jax.Array,
) -> jax.Array:
    """SPR loss: L2-normalized squared distance, masked by same_trajectory.

    Matches reference: ||norm(pred) - norm(target)||^2 summed over F, masked by trajectory.

    Returns per-sample SPR loss (B,).
    """
    # Cast to fp32: eps=1e-6 is below bf16 ULP (~1e-3 at magnitude 1), and the
    # F=2048 sum accumulates error in bf16.
    predictions_JBF = predictions_JBF.astype(jnp.float32)
    targets_JBF = targets_JBF.astype(jnp.float32)
    pred_norm_JBF = predictions_JBF / (jnp.linalg.norm(predictions_JBF, axis=-1, keepdims=True) + 1e-6)
    tgt_norm_JBF = targets_JBF / (jnp.linalg.norm(targets_JBF, axis=-1, keepdims=True) + 1e-6)
    sq_dist_JB = jnp.sum((pred_norm_JBF - tgt_norm_JBF) ** 2, axis=-1)  # (J, B)
    masked_JB = sq_dist_JB * same_traj_mask_JB  # (J, B)
    return masked_JB.mean(axis=0)  # (B,) — mean over jumps
