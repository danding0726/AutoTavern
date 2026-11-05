"""Evaluate a policy or uniform legal-action baseline on explicit, independent seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import taverngym
from taverngym.policy import load_checkpoint, observation_tensors
from taverngym.train import positive_int


@torch.no_grad()
def evaluate(model, games, seed, config, device, sample=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    placements, lengths, rewards = [], [], []
    env = taverngym.BattlegroundsEnv(seed=seed, config=config)
    if model is not None:
        model.eval()
    for episode in range(games):
        obs, _ = env.reset(seed=seed + episode)
        done, length, total_reward = False, 0, 0.0
        while not done:
            if model is None:
                action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            else:
                distribution, _ = model(*observation_tensors(obs, device))
                action = int((distribution.sample() if sample
                              else distribution.logits.argmax(dim=-1)).item())
            obs, reward, terminated, truncated, info = env.step(action)
            if info["illegal"] is not None:
                raise RuntimeError(f"Illegal evaluation action: {info['illegal']}")
            done = terminated or truncated
            total_reward += reward
            length += 1
        if not 1 <= info["placement"] <= 8:
            raise RuntimeError("The simulator stopped without a valid placement")
        placements.append(info["placement"])
        lengths.append(length)
        rewards.append(total_reward)
    places = np.array(placements)
    return {
        "games": games,
        "seed": seed,
        "policy": "random" if model is None else ("sample" if sample else "greedy"),
        "env_config": config,
        "mean_placement": float(places.mean()),
        "placement_standard_error": (float(places.std(ddof=1) / np.sqrt(games))
                                     if games > 1 else None),
        "top4_rate": float(np.mean(places <= 4)),
        "win_rate": float(np.mean(places == 1)),
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "placement_counts": {str(p): int(np.sum(places == p)) for p in range(1, 9)},
        "placements": placements,
        "illegal_actions": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument("--random", action="store_true", help="Uniform over legal actions")
    parser.add_argument("--games", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--action-cap", type=positive_int, default=60)
    parser.add_argument("--sample", action="store_true",
                        help="Sample instead of greedy policy actions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    parser.add_argument("--out", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    config = {"turn_action_caps": [args.action_cap]}
    model = None
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        if checkpoint["env_config"] != config:
            parser.error("--action-cap must match the checkpoint")
    result = evaluate(model, args.games, args.seed, config, device, args.sample)
    if args.checkpoint:
        result["checkpoint"] = str(args.checkpoint)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
