# Changelog

## [0.1.0] - 2026-04-14

Initial release of nexrl, a pure-JAX RL library.

### Algorithms

- On-policy: PPO, PPO-RNN, PPG, PPO-RND, PPO-ICM, PPO-E3B, V-MPO, PQN
- Off-policy: SAC, SAC-SimBa, TD3, DQN, IQN, Rainbow, MPO, DrQ-v2
- Model-based: Dreamer, MuZero, AlphaZero, MrQ, MRS.Q, SGF

### Features

- All algorithms are fully `jax.jit` and `jax.vmap` compatible
- Typed dataclass configs with YAML support and CLI overrides via tyro
- Flax NNX for all neural network components
- W&B integration for post-training upload
- Environment adapters for Gymnax, Brax, Craftax, Atari, DM Control, MuJoCo Playground, and more
