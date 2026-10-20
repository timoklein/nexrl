"""Typed configuration dataclasses for nexrl RL algorithms."""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """Network architecture configuration."""

    activation: str = "swish"
    hidden_layer_sizes: tuple[int, ...] = (64, 64)


@dataclasses.dataclass(frozen=True)
class CNNConfig:
    """CNN encoder configuration for image-based observations."""

    encoder: str = "nature"  # "nature" or "impala"
    features_dim: int = 512  # 512 for nature, 256 for impala
    width_scale: int = 1  # channel multiplier for IMPALA encoder
    # Conv padding (nature encoder only; IMPALA is SAME by design). "VALID" is the
    # Nature-DQN convention (84x84 -> 7x7x64 = 3136 flat). "SAME" matches Dopamine's
    # FullRainbowNetwork, whose nn.Conv calls take the Flax default (84x84 -> 11x11x64
    # = 7744 flat) — the network that produced the published DrQ Atari-100k numbers.
    padding: str = "VALID"


def _dict_to_dataclass(cls: type, d: dict[str, Any]) -> Any:
    """Recursively construct a dataclass from a plain dict."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in d:
            continue
        val = d[field.name]
        field_type = hints.get(field.name)
        # Resolve dataclass type — handles plain type and Union (e.g., CNNConfig | None)
        dc_type = None
        if dataclasses.is_dataclass(field_type) and isinstance(field_type, type):
            dc_type = field_type
        elif isinstance(val, dict):
            for arg in getattr(field_type, "__args__", ()):
                if dataclasses.is_dataclass(arg) and isinstance(arg, type):
                    dc_type = arg
                    break
        if dc_type is not None and isinstance(val, dict):
            val = _dict_to_dataclass(dc_type, val)
        elif isinstance(val, list):
            # YAML loads sequences as lists; convert to tuple when field type expects it
            is_tuple = getattr(field_type, "__origin__", None) is tuple
            if not is_tuple and field_type is not None:
                # Handle Union types (e.g., tuple[int, ...] | None)
                args = getattr(field_type, "__args__", ())
                is_tuple = any(getattr(a, "__origin__", a) is tuple for a in args)
            if is_tuple:
                val = tuple(val)
        kwargs[field.name] = val
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class _BaseConfig:
    """Base config with from_yaml support."""

    encoder_kwargs: CNNConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> _BaseConfig:
        """Load config from YAML, using file values as defaults."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return _dict_to_dataclass(cls, d)

    def validate(self) -> None:
        """Warn about common misconfigurations."""
        if self.encoder_kwargs is not None and getattr(self, "normalize_observations", False):
            warnings.warn(
                "normalize_observations=True with image encoder is not recommended. "
                "Running mean/variance normalization on pixel values is meaningless. "
                "Set normalize_observations=False.",
                stacklevel=2,
            )


@dataclasses.dataclass(frozen=True)
class PPOConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 64
    num_steps: int = 64
    num_minibatches: int = 16
    num_epochs: int = 8
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2  # PPO clipping range for policy ratio
    ent_coef: float = 0.01  # entropy bonus coefficient
    vf_coef: float = 0.5  # value loss coefficient
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False
    shared_backbone: bool = False  # share encoder between actor and critic
    map_obs_shape: tuple[int, ...] | None = None  # reshape obs before network (e.g. Craftax)
    layer_width: int = 512  # width for shared_backbone encoder
    adam_eps: float = 1e-5
    anneal_lr: bool = False  # linearly decay LR to 0 over training


@dataclasses.dataclass(frozen=True)
class PPORNNConfig(_BaseConfig):
    """PPO-RNN config — fixed architecture with layer_size instead of agent_kwargs.

    Matches the Craftax baselines reference (MichaelTMatthews/craftax_baselines).
    """

    env: str = "CartPole-v1"
    layer_size: int = 512
    num_envs: int = 1024
    num_steps: int = 64
    num_minibatches: int = 8
    num_epochs: int = 4
    learning_rate: float = 0.0002
    max_grad_norm: float = 1.0
    total_timesteps: int = 1_000_000_000
    eval_freq: int = 1_000_000
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    adam_eps: float = 1e-5
    anneal_lr: bool = True
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class PPORNDConfig(PPOConfig):
    intrinsic_reward_coef: float = 1.0  # weight of intrinsic vs extrinsic reward
    rnd_embedding_dim: int = 64
    rnd_hidden_layer_sizes: tuple[int, ...] = (256, 256)
    intrinsic_gamma: float = 0.99  # discount for non-episodic intrinsic value
    intrinsic_reward_norm: bool = False  # running-mean normalize intrinsic rewards
    rnd_learning_rate: float = 3e-4  # separate optimizer for RND predictor
    rnd_update_epochs: int = 1  # single epoch prevents predictor overfitting


@dataclasses.dataclass(frozen=True)
class PPOICMConfig(PPOConfig):
    icm_latent_size: int = 64  # ICM encoder output dimension
    icm_hidden_layer_sizes: tuple[int, ...] = (256, 256)
    intrinsic_reward_coef: float = 1.0  # weight of intrinsic vs extrinsic reward
    icm_forward_loss_coef: float = 1.0  # forward model prediction loss weight
    icm_inverse_loss_coef: float = 1.0  # inverse model action prediction loss weight
    icm_loss_coef: float = 1.0  # overall ICM loss scale
    intrinsic_gamma: float = 0.99  # discount for non-episodic intrinsic value
    intrinsic_reward_norm: bool = True  # running-mean normalize intrinsic rewards


@dataclasses.dataclass(frozen=True)
class PPOE3BConfig(PPOICMConfig):
    e3b_lambda: float = 0.1  # inverse covariance regularization
    e3b_reward_coef: float = 1.0  # elliptical bonus scale


@dataclasses.dataclass(frozen=True)
class PPGConfig(PPOConfig):
    n_policy_iters: int = 32  # PPO iterations between auxiliary phases
    num_aux_epochs: int = 6  # epochs in auxiliary phase
    aux_num_minibatches: int = 16
    beta_clone: float = 1.0  # KL regularization weight during auxiliary phase


@dataclasses.dataclass(frozen=True)
class SACConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99  # Polyak averaging coefficient for target networks
    target_update_freq: int = 1  # steps between target network updates
    target_entropy_ratio: float | None = None  # target entropy as ratio of -log(1/|A|); None = auto
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class SimBaNetworkConfig:
    """SimBa network architecture configuration."""

    hidden_dim: int = 256  # residual block hidden dimension
    num_blocks: int = 2  # number of residual feedforward blocks


@dataclasses.dataclass(frozen=True)
class SACSimBaConfig(_BaseConfig):
    env: str = "Pendulum-v1"
    actor_kwargs: SimBaNetworkConfig = dataclasses.field(
        default_factory=lambda: SimBaNetworkConfig(hidden_dim=128, num_blocks=1)
    )
    critic_kwargs: SimBaNetworkConfig = dataclasses.field(
        default_factory=lambda: SimBaNetworkConfig(hidden_dim=512, num_blocks=2)
    )
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 2
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-4
    actor_weight_decay: float = 1e-2
    critic_weight_decay: float = 1e-2
    initial_alpha: float = 0.01  # initial entropy temperature
    target_entropy_coef: float = -0.5  # target entropy = coef * action_dim
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.995
    target_update_freq: int = 1
    normalize_observations: bool = True
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class FastSACConfig(_BaseConfig):
    # Defaults: RL-X fastsac/flax_full_jit default_config + its G1 env config (4096 envs).
    env: str = "playground/G1JoystickFlatTerrain"
    obs_key: str | None = "privileged_state"  # dict-obs entry fed to actor and critic (Playground)
    num_envs: int = 4_096
    buffer_size_per_env: int = 1_024  # ring rows per env (total = buffer_size_per_env * num_envs)
    learning_starts: int = 10  # vectorized env steps (x num_envs transitions) before the first update
    batch_size: int = 8_192
    actor_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    critic_hidden_layer_sizes: tuple[int, ...] = (768, 384, 192)
    learning_rate: float = 3e-4  # actor, critic and alpha
    weight_decay: float = 1e-3  # AdamW, also applied to log_alpha
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    total_timesteps: int = 2_000_158_720
    eval_freq: int = 18_350_080
    gamma: float = 0.97
    polyak: float = 0.875  # reference tau=0.125, applied after every critic update
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    target_entropy: float = 0.0
    initial_alpha: float = 0.001
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    num_critic_updates_per_policy_update: int = 4
    num_policy_updates_per_step: int = 2  # critic updates per env step = 4 * 2 = 8
    normalize_observations: bool = True
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class FlashSACConfig(_BaseConfig):
    # Defaults: RL-X flashsac/flax_full_jit default_config + its G1 env config (4096 envs).
    # No observation normalization: the networks' input BatchNorm takes its place.
    env: str = "playground/G1JoystickFlatTerrain"
    obs_key: str | None = "privileged_state"  # dict-obs entry fed to actor and critic (Playground)
    num_envs: int = 4_096
    buffer_size: int = 10_000_000  # transitions in total; ring rows per env = buffer_size // num_envs
    learning_starts: int = 100_000  # timesteps of uniform random actions before the first update
    batch_size: int = 2_048
    updates_per_step: int = 2  # critic updates per vectorized env step
    policy_delay: int = 2  # policy + alpha update every policy_delay critic updates
    learning_rate: float = 3e-4  # cosine decay from here, for actor, critic and alpha
    learning_rate_end: float = 1.5e-4
    total_timesteps: int = 50_003_968  # reference 50_000_896 rounded up to a multiple of 4096 envs
    eval_freq: int = 6_250_496
    gamma: float = 0.99
    polyak: float = 0.99  # reference tau=0.01, applied after every critic update
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0
    initial_alpha: float = 0.01
    target_entropy_sigma: float = 0.15  # target entropy = entropy of a Gaussian with this std per action dim
    normalize_rewards: bool = True
    normalized_g_max: float = 5.0  # reward scale keeps |return| <= this
    noise_zeta_mu: float = 2.0  # exploration noise is held for n ~ zeta(mu) steps ...
    noise_zeta_max: int = 16  # ... truncated at noise_zeta_max
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class XQCConfig(_BaseConfig):
    # Defaults: RL-X xqc/flax default_config (one env, host loop in the reference).
    # No observation normalization: the networks' input BatchNorm takes its place.
    env: str = "playground/HumanoidRun"
    obs_key: str | None = None  # dict-obs entry fed to actor and critic (Playground); HumanoidRun obs is flat
    # Inner env steps per agent step (playground/, dm_control/). RL-X uses 1; the paper uses 2 on DMC.
    action_repeat: int = 1
    num_envs: int = 1
    buffer_size: int = 1_000_000  # transitions in total; ring rows per env = buffer_size // num_envs
    learning_starts: int = 5_000  # timesteps of uniform random actions; updates start once exceeded
    batch_size: int = 256
    updates_per_step: int = 2  # critic updates per vectorized env step
    policy_delay: int = 3  # policy + alpha step every policy_delay updates (policy BN stats every update)
    actor_learning_rate: float = 3e-4  # linear decay from each of these to learning_rate_end
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    learning_rate_end: float = 3e-5
    total_timesteps: int = 1_000_000
    eval_freq: int = 50_000
    gamma: float = 0.99
    polyak: float = 0.995  # reference tau=0.005, applied after every critic update
    actor_hidden_dim: int = 256
    actor_num_blocks: int = 4
    critic_hidden_dim: int = 512
    critic_num_blocks: int = 4
    num_critics: int = 2
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    initial_alpha: float = 0.01
    target_entropy: float | None = None  # None = reference "auto" = -action_dim / 2
    normalize_rewards: bool = True  # divide by the running std of the discounted return
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class REPPOConfig(_BaseConfig):
    # Defaults: RL-X reppo/flax_full_jit default_config + its G1 env config (4096 envs).
    env: str = "playground/G1JoystickFlatTerrain"
    obs_key: str | None = "privileged_state"  # dict-obs entry fed to critic and normalizer; policy sees policy_obs_dim dims
    policy_obs_dim: int | None = 103  # G1 `state` = first 103 dims of `privileged_state` (RL-X); None = full obs
    num_envs: int = 4_096
    num_steps: int = 128  # rollout length per env
    num_epochs: int = 4
    num_minibatches: int = 128
    learning_rate: float = 3e-4  # actor and critic
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_989_672_960  # reference 2e9 runs 115 eval periods of 33 rollouts; this matches that count
    eval_freq: int = 17_301_504  # 33 rollouts of 4096 x 128
    gamma: float = 0.99
    gae_lambda: float = 0.95  # TD(lambda) of the soft value targets
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 512
    policy_min_std: float = 0.0
    num_bins: int = 151  # HL-Gauss bins of the categorical critic
    v_min: float = -100.0
    v_max: float = 100.0
    init_kl_coefficient: float = 0.01
    kl_bound: float = 0.1
    init_entropy_coefficient: float = 0.01
    target_entropy_multiplier: float = 0.5  # the entropy coefficient drives the entropy toward -action_dim * this
    aux_loss_coefficient: float = 1.0  # next-feature and reward prediction loss
    num_kl_samples: int = 16  # old-policy action samples per state for the KL estimate
    normalize_observations: bool = True
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class FastTD3Config(_BaseConfig):
    # Defaults: RL-X fasttd3/flax_full_jit default_config + its G1 env config (4096 envs).
    env: str = "playground/G1JoystickFlatTerrain"
    obs_key: str | None = "privileged_state"  # dict-obs entry fed to actor and critic (Playground)
    num_envs: int = 4_096
    buffer_size_per_env: int = 10_240  # ring rows per env (total = buffer_size_per_env * num_envs)
    learning_starts: int = 10  # vectorized env steps (x num_envs transitions) before the first update
    batch_size: int = 32_768
    actor_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    critic_hidden_layer_sizes: tuple[int, ...] = (1024, 512, 256)
    learning_rate: float = 3e-4  # actor and critic
    weight_decay: float = 0.1  # AdamW
    total_timesteps: int = 2_000_158_720
    eval_freq: int = 18_350_080
    gamma: float = 0.97
    polyak: float = 0.9  # reference tau=0.1, applied after every critic update
    num_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    noise_std_min: float = 0.001  # per-env exploration noise std ~ U(min, max), resampled at episode end
    noise_std_max: float = 0.4
    smoothing_noise_std: float = 0.001  # target policy smoothing noise std ...
    smoothing_noise_clip: float = 0.5  # ... clipped to +-this
    num_critic_updates_per_policy_update: int = 2
    num_policy_updates_per_step: int = 1  # critic updates per env step = 2 * 1 = 2
    normalize_observations: bool = True
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class TD3Config(_BaseConfig):
    env: str = "Pendulum-v1"
    actor_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    critic_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    exploration_noise: float = 0.3  # stddev of Gaussian exploration noise
    target_noise: float = 0.2  # stddev of smoothing noise on target actions
    target_noise_clip: float = 0.5  # clipping range for target noise
    policy_delay: int = 2  # actor update frequency relative to critic
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class DQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent: str = "QNetwork"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    eps_start: float = 1.0  # initial epsilon for epsilon-greedy
    eps_end: float = 0.01  # final epsilon
    exploration_fraction: float = 0.5  # fraction of training for epsilon decay
    # Epsilon used by the eval policy. 0.001 is the Dopamine/DER canonical value (matches BBFConfig).
    epsilon_eval: float = 0.001
    ddqn: bool = True  # use Double DQN for target computation
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class DrQConfig(_BaseConfig):
    """DrQ (Kostrikov et al. 2021), discrete Atari-100k variant.

    Reference: google/dopamine labs/atari_100k DrQ.gin — Double + Dueling DQN,
    n-step 10, one random-shift + intensity augmentation per sampled obs/next_obs.
    """

    env: str = "atari/Breakout-v5"
    agent: str = "DuelingQNetwork"  # DrQ.gin: dueling = True
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1  # Dopamine runs a single env
    # 110000 holds all 100k transitions of an Atari-100k run without wrapping — identical
    # training to the gin's 1M capacity, without the OOM (see configs/bbf/atari/breakout.yaml).
    buffer_size: int = 110_000
    fill_buffer: int = 1_600  # min_replay_history
    batch_size: int = 32
    num_epochs: int = 1  # update_period=1, num_updates_per_train_step=1 -> replay ratio 1
    learning_rate: float = 0.0001
    adam_eps: float = 1.5e-4  # Dopamine create_optimizer eps (NOT the optax 1e-8 default)
    max_grad_norm: float = 10.0  # paper Table 4 "Max gradient norm 10" (global-norm clip)
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    gamma: float = 0.99
    n_step: int = 10  # update_horizon
    # target_update_freq=1 takes the polyak branch of maybe_update_targets; polyak=0.0
    # degenerates it to a HARD copy every env step == Dopamine target_update_period=1
    # at replay ratio 1. Tuning polyak away from 0.0 silently breaks reference fidelity.
    polyak: float = 0.0
    target_update_freq: int = 1
    epsilon_train: float = 0.1
    epsilon_eval: float = 0.05
    epsilon_decay_period: int = 5_000  # env steps of linear decay AFTER buffer fill
    ddqn: bool = True  # DrQ.gin: double_dqn = True
    # False: NNX default init (lecun_normal everywhere). True: xavier_uniform for EVERY
    # conv/dense (encoder + MLP + dueling heads) — the reference default: Dopamine's
    # FullRainbowNetwork uses nn.initializers.xavier_uniform() on every layer.
    xavier_init: bool = False
    # The ONE DrQ augmentation: random shift (edge-pad shift_pad, per-image offset in
    # [0, 2*shift_pad) — Dopamine _per_image_random_crop) then per-image intensity noise.
    data_augmentation: bool = True
    shift_pad: int = 4
    intensity_scale: float = 0.05
    terminal_on_life_loss: bool = False  # set true for Atari; create_env plumbs it
    # Random no-op starts per reset (gymnasium AtariPreprocessing noop_max; train AND
    # eval). 30 is the gymnasium default and what all runs so far used. The Dopamine
    # reference applies NO no-ops anywhere — its env is fully deterministic — so 0 is
    # the reference-faithful value (A/B pending).
    noop_max: int = 30
    clip_rewards: bool = True  # clip training rewards to [-1, 1]; metrics stay raw
    # Evaluation protocol. `eval_episodes` complete episodes are always played (every
    # slot a real episode — no zero-fill deflation). `eval_step_budget` additionally
    # reports the PAPER's estimator: Table 4 "Evaluation steps 125k" means Dopamine's
    # _run_one_phase, which plays complete episodes until the cumulative agent-step
    # count reaches the budget, so the episode COUNT is variable. 0 disables it (the
    # reported paper metric then equals the plain mean over all eval_episodes).
    eval_episodes: int = 128
    eval_step_budget: int = 0
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class IQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 1
    num_epochs: int = 1
    num_tau_samples: int = 64  # quantile samples for online network
    num_tau_prime_samples: int = 64  # quantile samples for target network
    kappa: float = 1.0  # Huber loss threshold for quantile regression
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.01
    exploration_fraction: float = 0.5
    # Epsilon used by the eval policy. 0.001 is the Dopamine/DER canonical value (matches BBFConfig).
    epsilon_eval: float = 0.001
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class PQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    # PQN create_agent defaults to relu when agent_kwargs not given
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=lambda: NetworkConfig(activation="relu"))
    num_envs: int = 64
    num_steps: int = 64
    num_minibatches: int = 16
    num_epochs: int = 1
    td_lambda: float = 0.9  # lambda for TD(lambda) returns
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.1
    # Epsilon used by the eval policy. 0.001 is the Dopamine/DER canonical value (matches BBFConfig).
    epsilon_eval: float = 0.001
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False
    norm_type: str = "none"  # normalization layer: "none", "layer_norm", "batch_renorm"
    norm_input: bool = False  # normalize network input


@dataclasses.dataclass(frozen=True)
class RainbowConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = 10.0
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.01
    exploration_fraction: float = 0.5
    # Epsilon used by the eval policy. 0.001 is the Dopamine/DER canonical value (matches BBFConfig).
    epsilon_eval: float = 0.001
    # C51
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    # N-step
    n_step: int = 3
    # PER
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6
    # Standard flags
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class BBFConfig(_BaseConfig):
    """BBF — Bigger, Better, Faster (Schwarzer et al., ICML 2023).

    Builds on Rainbow (C51 + PER + dueling + double DQN) with SPR auxiliary loss,
    scaled IMPALA encoder, periodic shrink-and-perturb resets, receding n-step/gamma
    annealing, EMA target network, and AdamW weight decay.
    """

    env: str = "atari/Breakout-v5"
    # Collection
    num_envs: int = 1
    buffer_size: int = 200_000
    fill_buffer: int = 2_000
    batch_size: int = 32
    num_epochs: int = 8  # replay ratio = num_epochs * num_envs
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    # C51
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    # N-step annealing (per-cycle: max_n_step -> n_step over cycle_steps grad steps)
    n_step: int = 3
    max_n_step: int = 10
    # Gamma annealing (per-cycle: min_gamma -> gamma)
    gamma: float = 0.997
    min_gamma: float = 0.97
    # Reset schedule
    cycle_steps: int = 10_000
    reset_every: int = 20_000
    no_resets_after: int = 100_000
    # Shrink-and-perturb
    shrink_factor: float = 0.5
    perturb_factor: float = 0.5
    # SPR
    spr_weight: float = 5.0
    jumps: int = 5
    # Network — hidden_dim is the encoder output / SPR projection dim. It is
    # used as features_dim when no CNN encoder is configured (testing path).
    hidden_dim: int = 2048
    # Mixed precision: "bfloat16" or "float32". Params (master weights, optimizer
    # state, LayerNorm, softmax/log_softmax, SPR norms, Polyak EMA) stay fp32;
    # only conv/linear forward passes run in compute_dtype.
    compute_dtype: str = "bfloat16"
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 10.0
    # Target network
    target_update_tau: float = 0.005
    target_action_selection: bool = True
    # Exploration
    epsilon_train: float = 0.0
    epsilon_eval: float = 0.001
    epsilon_decay_period: int = 2_000
    # Augmentation
    data_augmentation: bool = True
    shift_pad: int = 4
    # Random-shift offset range. Schwarzer's BBF crops with offset in [0, 2*pad)
    # (drq_image_aug -> _per_image_random_crop, randint(0, padded - cropped = 2*pad);
    # spr_networks.py:106-118), giving shifts in BOTH directions. Our shared
    # `random_shift` defaults to a one-sided [0, pad] range (SGF/DrQ-v2). True ->
    # pass the reference [0, 2*pad) range at BBF's augmentation call sites; False ->
    # the legacy one-sided range (ablation control). Only active when data_augmentation.
    full_range_shift: bool = True
    # Augment the SPR target's future frames (random-shift + intensity), exactly
    # like the current/next obs. Schwarzer's BBF augments the WHOLE subsequence
    # (spr_agent.py:587) then slices the augmented future frames for the SPR
    # target (spr_agent.py:725), so the target sees augmented views. Set False to
    # feed raw future frames (ablation control). Only active when data_augmentation.
    augment_spr_targets: bool = True
    # Terminal-on-life-loss (Atari): the agent sees a per-life terminal (n-step cut at
    # life boundaries) while the game continues across lives and train_return/eval stay
    # full-game. Matches Schwarzer's BBF runner. Off by default for non-Breakout configs.
    terminal_on_life_loss: bool = False
    # Reward clipping to [-1, 1] for the TRAINING signal (buffer/targets), matching
    # Schwarzer's runner. train_return is reported on the RAW reward (the reference
    # reports unclipped cum_reward), so the metric stays the true game score.
    clip_rewards: bool = True
    # PER
    per_alpha: float = 0.5
    per_beta: float = 0.5
    per_eps: float = 1e-6
    # Sharp per-sample PER (ours) vs the reference's EFFECTIVE near-uniform sampling.
    # True (default): every sampled transition's priority is updated to its own
    # (ce+eps)^alpha each step -> sampling concentrates on high-TD transitions, IS
    # weights vary strongly. False: replicate Schwarzer's *effective* behavior — his
    # set_priority zip-bug (spr_agent.py:1646-1649 zips (B*G,) indices with (G,)
    # batch-MEAN priorities) updates only ~2 of 64 sampled transitions per step, so
    # ~62/64 stay frozen at their max-priority insert value (subsequence_replay_buffer.py
    # _add: Schaul max-for-new). Net: ≈ uniform sampling + weak recency (new data at the
    # running max), IS weights ≈ 1. We skip the per-sample priority updates (freeze at
    # insert) but keep max_priority tracking the running-mean loss so new inserts sit at
    # the highest priority. Not the literal zip bug — its effect. The 357 agent ran False.
    sharp_per: bool = True
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False
    # When True, run the (expensive, sequential-Atari) 128-episode evaluation ONLY at
    # the final eval point instead of at every eval_freq boundary. The train_return
    # curve is still logged at every eval point; intermediate eval arrays are zeros.
    # Eval consumes RNG only at the final point, so training dynamics are identical to
    # a run with run_evaluation=False up to the last step.
    final_eval_only: bool = False
    # Initialization scheme. False (default): Flax NNX defaults (lecun_normal kernels
    # everywhere). True: xavier_uniform for EVERY conv/dense (encoder, transition convs,
    # projection, predictor, value/advantage heads) — matching Schwarzer's reference
    # (spr_networks.py initializer_type='xavier_uniform'). lecun/xavier std ratio is
    # ~2.9x (first conv) to ~0.75x (projection/heads); the init also re-seeds every S&P
    # reset (~18x/run), so it compounds. Audit finding #7.
    xavier_init: bool = False
    # CNN encoder
    encoder_kwargs: CNNConfig | None = dataclasses.field(
        default_factory=lambda: CNNConfig(encoder="impala", features_dim=2048, width_scale=4)
    )


@dataclasses.dataclass(frozen=True)
class MPOConfig(_BaseConfig):
    env: str = "CartPole-v1"
    actor_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    critic_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 1
    learning_rate: float = 0.0003
    critic_learning_rate: float = 0.0003
    dual_learning_rate: float = 0.01  # LR for Lagrange dual variables
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.995
    target_update_freq: int = 1
    # MPO E-step / M-step
    epsilon: float = 0.1  # E-step temperature constraint (KL bound)
    epsilon_policy: float = 0.01  # M-step KL constraint on policy update
    epsilon_mean: float = 1e-3  # per-dim KL bound on mean (continuous only)
    epsilon_stddev: float = 1e-6  # per-dim KL bound on stddev (continuous only)
    per_dim_constraining: bool = True  # separate KL constraints per action dimension
    num_action_samples: int = 20  # samples for E-step importance weighting
    init_log_temperature: float = 1.0  # initial log(eta) for E-step temperature
    init_log_alpha: float = 1.0  # initial log(alpha) for M-step KL dual
    init_log_alpha_mean: float = 10.0  # initial dual for per-dim mean constraint
    init_log_alpha_stddev: float = 1000.0  # initial dual for per-dim stddev constraint
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class VMPOConfig(_BaseConfig):
    env: str = "CartPole-v1"
    actor_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    critic_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 64
    num_steps: int = 64
    num_minibatches: int = 16
    num_epochs: int = 10
    learning_rate: float = 0.0003
    critic_learning_rate: float = 0.0003
    dual_learning_rate: float = 0.01
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    # V-MPO E-step / M-step
    epsilon: float = 0.1  # E-step temperature constraint (KL bound)
    epsilon_policy: float = 0.01  # M-step KL constraint on policy update
    epsilon_mean: float = 1e-3  # per-dim KL bound on mean (continuous only)
    epsilon_stddev: float = 1e-6  # per-dim KL bound on stddev (continuous only)
    per_dim_constraining: bool = True  # separate KL constraints per action dimension
    top_k_fraction: float = 0.5  # fraction of samples used in E-step advantage weighting
    actor_target_period: int = 10  # steps between hard target actor copies
    init_log_temperature: float = 1.0  # initial log(eta) for E-step temperature
    init_log_alpha: float = 1.0  # initial log(alpha) for M-step KL dual
    init_log_alpha_mean: float = 10.0  # initial dual for per-dim mean constraint
    init_log_alpha_stddev: float = 1000.0  # initial dual for per-dim stddev constraint
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class AlphaZeroConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 50_000
    fill_buffer: int = 500
    batch_size: int = 32
    num_epochs: int = 8
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    num_simulations: int = 25  # MCTS simulations per move
    max_depth: int | None = None  # MCTS max depth; None = unlimited
    mcts_policy: str = "muzero"  # mctx search: "muzero" (PUCT) or "gumbel"
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class MuZeroConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 50_000
    fill_buffer: int = 500
    batch_size: int = 32
    num_epochs: int = 8
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    num_simulations: int = 25  # MCTS simulations per move
    max_depth: int | None = None  # MCTS max depth; None = unlimited
    mcts_policy: str = "muzero"  # mctx search: "muzero" (PUCT) or "gumbel"
    # MuZero-specific
    latent_dim: int = 64  # learned representation dimension
    unroll_steps: int = 5  # K-step unrolling for dynamics loss
    n_steps: int = 10  # n-step returns for value targets
    reward_coef: float = 1.0  # reward prediction loss weight
    gradient_scale: float = 0.5  # gradient scaling on latent states
    # Categorical value/reward support (signed_hyperbolic transform)
    value_min: float = -300.0
    value_max: float = 300.0
    value_num_atoms: int = 601
    reward_min: float = -300.0
    reward_max: float = 300.0
    reward_num_atoms: int = 601
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class RSSMConfig:
    """RSSM architecture configuration for Dreamer-v3."""

    deter_dim: int = 512  # GRU deterministic state dimension
    stoch_categories: int = 32  # number of categorical distributions
    stoch_classes: int = 32  # classes per categorical distribution
    hidden_dim: int = 512  # MLP hidden dimension within RSSM
    unimix: float = 0.01  # uniform mixing to prevent categorical collapse


@dataclasses.dataclass(frozen=True)
class DreamerConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    rssm_kwargs: RSSMConfig = dataclasses.field(default_factory=RSSMConfig)
    # Collection
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 16
    sequence_length: int = 64
    # Imagination
    imagination_horizon: int = 15  # latent rollout length for actor-critic
    # Training
    num_epochs: int = 1
    learning_rate: float = 1e-4
    max_grad_norm: float = 100.0
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    # RL
    gamma: float = 0.997
    lambda_gae: float = 0.95
    # Loss scales
    recon_scale: float = 1.0  # observation reconstruction loss weight
    reward_scale: float = 1.0  # reward prediction loss weight
    continue_scale: float = 1.0  # continue predictor loss weight
    dyn_scale: float = 1.0  # dynamics (prior → posterior) KL weight
    rep_scale: float = 0.1  # representation (posterior → prior) KL weight
    policy_scale: float = 1.0
    value_scale: float = 1.0
    # World model
    kl_free_bits: float = 1.0  # free nats below which KL loss is zeroed
    # Actor
    ent_coef: float = 3e-4  # entropy bonus for policy in imagination
    return_norm_rate: float = 0.01  # EMA rate for percentile return normalization
    return_norm_limit: float = 1.0  # max scale for return normalization
    return_norm_perclo: float = 5.0  # lower percentile for return normalization
    return_norm_perchi: float = 95.0  # upper percentile for return normalization
    # Critic
    slow_value_rate: float = 0.02  # Polyak rate for slow critic (regularization target)
    slow_value_reg: float = 1.0  # weight of slow critic regularization loss
    # Symlog two-hot
    value_num_bins: int = 255  # bins for symlog two-hot value distribution
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class MrQConfig(_BaseConfig):
    env: str = "CartPole-v1"
    # Collection
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 10_000
    batch_size: int = 256
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    # Network dims
    hidden_dim: int = 512
    zs_dim: int = 512  # state embedding dimension
    za_dim: int = 256  # action embedding dimension
    zsa_dim: int = 512  # state-action embedding dimension
    activation: str = "elu"
    # Encoder (burst-updated every target_update_freq steps)
    enc_learning_rate: float = 1e-4
    enc_weight_decay: float = 1e-4
    enc_horizon: int = 5  # multi-step encoder prediction horizon
    dyn_weight: float = 1.0  # dynamics prediction loss weight
    reward_weight: float = 0.1  # reward prediction loss weight
    done_weight: float = 0.1  # termination prediction loss weight
    # Value
    value_learning_rate: float = 3e-4
    value_weight_decay: float = 1e-4
    value_grad_clip: float = 20.0
    q_horizon: int = 3  # n-step return horizon for Q-learning
    # Policy (gradients flow through encoder.zsa)
    policy_learning_rate: float = 3e-4
    policy_weight_decay: float = 1e-4
    pre_activ_weight: float = 1e-5  # L2 penalty on pre-tanh activations
    # Exploration
    exploration_noise: float = 0.2  # Gaussian noise stddev (scaled 0.5x for discrete)
    policy_activation: str = "relu"
    target_noise: float = 0.2
    target_noise_clip: float = 0.3
    gumbel_tau: float = 10.0  # Gumbel-softmax temperature (discrete actions)
    # Target updates (burst pattern)
    target_update_freq: int = 250  # steps between hard target copies + burst encoder updates
    # PER
    per_alpha: float = 0.4  # prioritized replay exponent
    min_priority: float = 1.0  # minimum priority floor
    # Reward bins (symexp two-hot)
    reward_num_bins: int = 65
    reward_lower: float = -10.0
    reward_upper: float = 10.0
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class MrsQConfig(_BaseConfig):
    env: str = "CartPole-v1"
    # Collection
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 10_000
    batch_size: int = 256
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    # Network dims
    hidden_dim: int = 512
    zs_dim: int = 512
    za_dim: int = 256
    zsa_dim: int = 512
    activation: str = "elu"
    num_critics: int = 10  # Q-ensemble size (target uses min)
    sem_groups: int = 8  # SEM softmax groups (requires zs_dim % sem_groups == 0)
    # Encoder
    enc_learning_rate: float = 1e-4
    enc_weight_decay: float = 1e-4
    enc_horizon: int = 5
    dyn_weight: float = 20.0
    reward_weight: float = 0.1
    done_weight: float = 1.0
    # Value
    value_learning_rate: float = 3e-4
    value_weight_decay: float = 1e-4
    value_grad_clip: float = 20.0
    q_horizon: int = 3
    # Policy
    policy_learning_rate: float = 3e-4
    policy_weight_decay: float = 1e-4
    pre_activ_weight: float = 1e-5
    # Exploration (MPPI replaces policy+noise)
    exploration_noise: float = 0.0
    policy_activation: str = "relu"
    target_noise: float = 0.2
    target_noise_clip: float = 0.3
    gumbel_tau: float = 10.0
    # Target updates (burst pattern)
    target_update_freq: int = 250
    # PER
    per_alpha: float = 0.4
    min_priority: float = 1.0
    # Reward bins
    reward_num_bins: int = 65
    reward_lower: float = -10.0
    reward_upper: float = 10.0
    # MPPI planning (replaces policy+noise for collection)
    mpc_horizon: int = 3  # planning horizon
    num_samples: int = 512  # total CEM trajectories per step
    num_pi_trajs: int = 24  # of num_samples, how many use policy mean
    num_elites: int = 64  # top-k for Gumbel-softmax elite selection
    mpc_iterations: int = 6  # CEM refinement iterations
    mpc_temperature: float = 0.5  # Gumbel-softmax temperature for elite selection
    min_std: float = 0.05  # minimum action distribution std
    max_std: float = 2.0  # maximum action distribution std
    pi_std: float = 0.1  # noise std for policy-seeded trajectories
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class SGFConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(
        default_factory=lambda: NetworkConfig(activation="silu", hidden_layer_sizes=(512, 512))
    )
    encoder_kwargs: CNNConfig | None = None
    # Collection
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 5_000
    batch_size: int = 1024
    imagination_batch_size: int = 3072  # batch size for imagination rollouts
    # Imagination
    imagination_horizon: int = 10  # latent rollout length for actor-critic
    # Training schedule
    wm_every: int = 2  # world model update frequency (env steps)
    agent_every: int = 2  # actor-critic update frequency (env steps)
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    # Representation optimizer
    repr_learning_rate: float = 6e-4
    repr_weight_decay: float = 1e-3
    repr_warmup_steps: int = 5000
    repr_grad_clip: float = 10.0
    # Transition optimizer
    transition_learning_rate: float = 1.2e-3
    transition_weight_decay: float = 1e-3
    transition_warmup_steps: int = 5000
    transition_grad_clip: float = 500.0
    # Actor-critic optimizer
    ac_learning_rate: float = 2.4e-6
    ac_grad_clip: float = 100.0
    # RL
    gamma: float = 0.997
    lambda_gae: float = 0.95
    # VICReg (self-supervised representation loss)
    consistency_weight: float = 12.5  # invariance term (augmented views should match)
    var_weight: float = 25.0  # variance term (prevent collapse)
    cov_weight: float = 1.0  # covariance term (decorrelate dimensions)
    # Loss scales
    reward_weight: float = 1.0
    terminal_weight: float = 1.0
    # Actor
    ent_coef: float = 1e-3
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0
    return_norm_perclo: float = 5.0
    return_norm_perchi: float = 95.0
    # Critic
    slow_value_rate: float = 0.02
    slow_value_reg: float = 1.0
    value_num_bins: int = 255
    reward_num_bins: int = 255
    # SGF-specific dims
    y_dim: int = 512  # encoder output dimension (with LayerNorm)
    z_dim: int = 2048  # projector output dimension (VICReg space)
    projector_hidden: int = 2048
    predictor_hidden: int = 2048
    transition_hidden: int = 1024  # transition predictor MLP width
    transition_num_hidden: int = 5  # transition predictor MLP depth
    # Augmentation (image-based only)
    shift_pad: int = 3  # random shift padding (pixels)
    intensity_scale: float = 0.05  # random intensity jitter scale
    # Action stacking
    action_stack: int = 4  # number of past actions concatenated to state
    # Exploration
    env_epsilon: float = 0.01  # epsilon-greedy during collection
    eval_temperature: float = 0.5  # softmax temperature for evaluation
    # Standard
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


@dataclasses.dataclass(frozen=True)
class DrQV2Config(_BaseConfig):
    env: str = "dm_control/cartpole-balance"
    feature_dim: int = 50  # encoder feature dimension
    hidden_layer_sizes: tuple[int, ...] = (1024, 1024)
    num_envs: int = 1
    buffer_size: int = 1_000_000
    seed_frames: int = 4000  # random exploration steps before training
    batch_size: int = 256
    n_step: int = 3  # n-step returns (computed at collection time)
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    update_every_steps: int = 2  # gradient steps per env step
    num_epochs: int = 1
    learning_rate: float = 1e-4
    max_grad_norm: float = float("inf")
    gamma: float = 0.99
    polyak: float = 0.99  # Polyak rate for critic target only (no target encoder)
    stddev_init: float = 1.0  # initial exploration noise stddev
    stddev_final: float = 0.1  # final exploration noise stddev
    stddev_schedule_steps: int = 500_000  # linear noise decay duration
    noise_clip: float = 0.3
    action_repeat: int = 2  # env action repeat (frame skip)
    frame_stack: int = 3  # stacked observation frames
    shift_pad: int = 4  # random shift augmentation padding (pixels)
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False
    run_evaluation: bool = False


ALGO_CONFIG_MAP: dict[str, type[_BaseConfig]] = {
    "alphazero": AlphaZeroConfig,
    "bbf": BBFConfig,
    "dreamer": DreamerConfig,
    "drq": DrQConfig,
    "drqv2": DrQV2Config,
    "mrq": MrQConfig,
    "mrsq": MrsQConfig,
    "muzero": MuZeroConfig,
    "sgf": SGFConfig,
    "ppo": PPOConfig,
    "ppo_rnn": PPORNNConfig,
    "ppg": PPGConfig,
    "ppo_rnd": PPORNDConfig,
    "ppo_icm": PPOICMConfig,
    "ppo_e3b": PPOE3BConfig,
    "sac": SACConfig,
    "sac_simba": SACSimBaConfig,
    "fast_sac": FastSACConfig,
    "flash_sac": FlashSACConfig,
    "xqc": XQCConfig,
    "reppo": REPPOConfig,
    "fast_td3": FastTD3Config,
    "td3": TD3Config,
    "dqn": DQNConfig,
    "iqn": IQNConfig,
    "pqn": PQNConfig,
    "rainbow": RainbowConfig,
    "mpo": MPOConfig,
    "mpo_continuous": MPOConfig,
    "vmpo": VMPOConfig,
    "vmpo_continuous": VMPOConfig,
}
