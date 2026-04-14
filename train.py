import argparse
import dataclasses
import functools
import timeit

import jax
import jax.numpy as jnp
import tyro
from matplotlib import pyplot as plt

from nexrl import ALGO_CONFIG_MAP, get_train_fn


_printed_steps: set[int] = set()


def _print_metrics(metrics):
    """Print one-line status at each eval point. Fires via jax.debug.callback.

    Under jax.vmap this fires once per seed. We only print the first seed's
    metrics by deduplicating on global_step.
    """
    step = int(metrics["global_step"])
    if step in _printed_steps:
        return
    _printed_steps.add(step)
    # Prefer train_return (from rollouts); fall back to eval_returns
    mean_ret = float(metrics["train_return"]) if "train_return" in metrics else float(metrics["eval_returns"].mean())
    parts = [f"step={step:>10,}", f"return={mean_ret:>8.1f}"]
    for k in sorted(metrics):
        if k.startswith("eval_") or k in ("global_step", "train_return"):
            continue
        v = metrics[k]
        if v.ndim == 0:
            parts.append(f"{k}={float(v):.4f}")
    print(" | ".join(parts), flush=True)


def main(
    algo_str: str,
    config: object,
    seed_id: int,
    num_seeds: int,
    time_fit: bool,
    quiet: bool,
    track: bool = False,
    wandb_project: str = "nexrl",
    wandb_entity: str | None = None,
) -> None:
    train_fn = get_train_fn(algo_str)
    if not quiet:
        train_fn = functools.partial(train_fn, log_fn=_print_metrics)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)), static_argnums=(0,))
    state, metrics = vmap_train(config, keys)
    returns = metrics["eval_returns"]
    returns.block_until_ready()

    if track:
        from datetime import datetime

        import wandb

        group_name = f"{algo_str}_{config.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        num_eval_points = returns.shape[1]
        global_steps = metrics["global_step"]  # (num_seeds, num_eval_points)

        for seed_idx in range(num_seeds):
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                group=group_name,
                name=f"seed_{seed_id + seed_idx}",
                config=dataclasses.asdict(config) | {"algorithm": algo_str, "seed_id": seed_id + seed_idx},
                reinit=True,
            )
            for step_idx in range(num_eval_points):
                log_dict = {"global_step": int(global_steps[seed_idx, step_idx])}
                for k, v in metrics.items():
                    if k == "global_step":
                        continue
                    val = v[seed_idx, step_idx]
                    if val.ndim == 0:
                        log_dict[k] = float(val)
                    elif val.ndim == 1:
                        log_dict[f"{k}/mean"] = float(val.mean())
                        log_dict[f"{k}/std"] = float(val.std())
                run.log(log_dict, step=int(global_steps[seed_idx, step_idx]))
            run.finish()

    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * config.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(config, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(f"{time:.1f} seconds total, equalling to {time / num_seeds:.1f} seconds per seed")

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())


if __name__ == "__main__":
    # First pass: extract training args and algorithm selection
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("algorithm", type=str, help="Algorithm name (ppo, sac, sac_simba, td3, dqn, iqn, pqn)")
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    pre_parser.add_argument("--seed-id", type=int, default=0, help="Random seed")
    pre_parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to roll out")
    pre_parser.add_argument("--time-fit", action="store_true", help="Time training by fitting 3 times")
    pre_parser.add_argument("--quiet", action="store_true", help="Suppress per-eval progress logging")
    pre_parser.add_argument("--track", action="store_true", help="Log to Weights & Biases")
    pre_parser.add_argument("--wandb-project", type=str, default="nexrl", help="wandb project name")
    pre_parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (team/user)")
    pre_args, remaining = pre_parser.parse_known_args()

    # Load YAML defaults and apply CLI overrides via tyro
    config_cls = ALGO_CONFIG_MAP[pre_args.algorithm]
    default = config_cls.from_yaml(pre_args.config) if pre_args.config else config_cls()
    config = tyro.cli(config_cls, default=default, args=remaining)

    main(
        pre_args.algorithm,
        config,
        pre_args.seed_id,
        pre_args.num_seeds,
        pre_args.time_fit,
        pre_args.quiet,
        pre_args.track,
        pre_args.wandb_project,
        pre_args.wandb_entity,
    )
