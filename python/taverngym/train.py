"""PPO against seven scripted bots: python -m taverngym.train --help."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import taverngym
from taverngym.policy import (
    Policy,
    completed_placements,
    load_checkpoint,
    observation_tensors,
    require_ready,
    save_checkpoint,
)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    """dones[t] ends transition t, including an automatic reset in BatchEnv."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        alive = (~dones[t]).to(rewards.dtype)
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * alive * following - values[t]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[t] = carry
    return advantages, advantages + values


@torch.no_grad()
def collect_rollout(env, model, obs, n_steps, device):
    n = env.n_slots
    shapes = {
        "features": ((n_steps, n, taverngym.OBS_FLOATS), torch.float32),
        "cards": ((n_steps, n, taverngym.OBS_CARDS), torch.long),
        "action_mask": ((n_steps, n, taverngym.N_ACTIONS), torch.bool),
        "actions": ((n_steps, n), torch.long),
        "log_probs": ((n_steps, n), torch.float32),
        "values": ((n_steps, n), torch.float32),
        "rewards": ((n_steps, n), torch.float32),
        "dones": ((n_steps, n), torch.bool),
    }
    rollout = {key: torch.empty(shape, dtype=dtype, device=device)
               for key, (shape, dtype) in shapes.items()}
    placements = []
    for t in range(n_steps):
        require_ready(obs)
        features, cards, mask = observation_tensors(obs, device)
        distribution, value = model(features, cards, mask)
        actions = distribution.sample()
        rewards, dones, info = env.step(actions.cpu().numpy())
        placements.extend(completed_placements(dones, info).tolist())
        rollout["features"][t].copy_(features)
        rollout["cards"][t].copy_(cards)
        rollout["action_mask"][t].copy_(mask)
        rollout["actions"][t].copy_(actions)
        rollout["log_probs"][t].copy_(distribution.log_prob(actions))
        rollout["values"][t].copy_(value)
        rollout["rewards"][t].copy_(torch.as_tensor(rewards, device=device))
        rollout["dones"][t].copy_(torch.as_tensor(dones, device=device))
        obs = env.observe()
    require_ready(obs)
    _, next_value = model(*observation_tensors(obs, device))
    return rollout, obs, next_value, placements


def ppo_update(model, optimizer, rollout, advantages, returns, epochs, minibatch_size,
               clip_coef=0.2, entropy_coef=0.01, value_coef=0.5):
    batch = {key: value.flatten(0, 1) for key, value in rollout.items()}
    advantages, returns = advantages.flatten(), returns.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = len(advantages)
    totals = np.zeros(5, dtype=np.float64)
    samples = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=advantages.device)
        for indices in order.split(minibatch_size):
            distribution, values = model(
                batch["features"][indices], batch["cards"][indices],
                batch["action_mask"][indices])
            log_ratio = (distribution.log_prob(batch["actions"][indices])
                         - batch["log_probs"][indices])
            ratio = log_ratio.exp()
            unclipped = -advantages[indices] * ratio
            clipped = -advantages[indices] * ratio.clamp(1 - clip_coef, 1 + clip_coef)
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = ((ratio - 1).abs() > clip_coef).float().mean()
                metrics = torch.stack((policy_loss, value_loss, entropy, kl, clip_fraction))
            totals += metrics.detach().cpu().numpy() * len(indices)
            samples += len(indices)
    return dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"),
                    (totals / samples).tolist()))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--resume", type=Path, help="Continue weights/optimizer; start fresh games")
    parser.add_argument("--updates", type=positive_int, default=1000, help="Additional PPO updates")
    parser.add_argument("--n-envs", type=positive_int, default=32)
    parser.add_argument("--rollout-steps", type=positive_int, default=256)
    parser.add_argument("--epochs", type=positive_int, default=4)
    parser.add_argument("--minibatch-size", type=positive_int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--action-cap", type=positive_int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=positive_int, default=4, help="Rust environment workers")
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if not (0 < args.gamma <= 1 and 0 <= args.gae_lambda <= 1
            and np.isfinite(args.lr) and args.lr > 0):
        parser.error("require 0 < gamma <= 1, 0 <= gae-lambda <= 1, lr > 0")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not args.resume and any((args.out / name).exists()
                               for name in ("latest.pt", "metrics.jsonl")):
        parser.error("Output already contains a run; use --resume or a new --out directory")
    device = torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    config = {"turn_action_caps": [args.action_cap]}
    start_update, steps = 0, 0
    torch.manual_seed(args.seed)
    model = Policy().to(device)
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["env_config"] != config:
            parser.error("--action-cap must match the checkpoint")
        start_update, steps = checkpoint["update"], checkpoint["steps"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    # Resume starts new games and a new exploration stream, not an exact simulator replay.
    run_seed = args.seed + start_update * args.n_envs
    torch.manual_seed(run_seed)
    env = taverngym.BatchEnv(args.n_envs, seed=run_seed, controlled=(0,),
                             config=config, threads=args.threads)
    obs = env.observe()
    recent = deque(maxlen=100)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    model.train()
    with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for update in range(start_update + 1, start_update + args.updates + 1):
            started = time.monotonic()
            rollout, obs, next_value, places = collect_rollout(
                env, model, obs, args.rollout_steps, device)
            advantages, returns = compute_gae(
                rollout["rewards"], rollout["dones"], rollout["values"], next_value,
                args.gamma, args.gae_lambda)
            metrics = ppo_update(model, optimizer, rollout, advantages, returns,
                                 args.epochs, args.minibatch_size)
            steps += args.n_envs * args.rollout_steps
            recent.extend(places)
            metrics.update({
                "update": update, "steps": steps, "episodes": len(places),
                "recent_episodes": len(recent),
                "mean_placement": float(np.mean(recent)) if recent else None,
                "top4_rate": float(np.mean(np.array(recent) <= 4)) if recent else None,
                "win_rate": float(np.mean(np.array(recent) == 1)) if recent else None,
                "steps_per_second": (args.n_envs * args.rollout_steps
                                     / (time.monotonic() - started)),
            })
            line = json.dumps(metrics, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            save_checkpoint(args.out / "latest.pt", model, optimizer, update, steps, config, args.seed)
    return metrics


if __name__ == "__main__":
    main()
