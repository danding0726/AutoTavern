"""Small masked actor-critic shared by PPO training and evaluation (requires torch)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import taverngym


def observation_tensors(obs, device):
    return (
        torch.as_tensor(obs["features"], dtype=torch.float32, device=device),
        torch.as_tensor(obs["cards"], dtype=torch.long, device=device),
        torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device),
    )


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(taverngym.N_CARDS, 16, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(taverngym.OBS_FLOATS + taverngym.OBS_CARDS * 16, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )
        self.actor = nn.Linear(256, taverngym.N_ACTIONS)
        self.critic = nn.Linear(256, 1)
        nn.init.normal_(self.actor.weight, std=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, features, cards, action_mask):
        if not action_mask.any(dim=-1).all():
            raise ValueError("Every policy observation must have a legal action")
        # Stats can grow far beyond the engine's nominal scaling in long games.
        features = features.sign() * features.abs().log1p()
        embedded = self.embedding(cards).flatten(start_dim=-2)
        encoded = self.encoder(torch.cat((features, embedded), dim=-1))
        logits = self.actor(encoded).masked_fill(~action_mask, torch.finfo(encoded.dtype).min)
        return Categorical(logits=logits), self.critic(encoded).squeeze(-1)


def engine_signature():
    return {
        "source_hash": taverngym.build_info()["source_hash"],
        "obs_floats": taverngym.OBS_FLOATS,
        "obs_cards": taverngym.OBS_CARDS,
        "n_cards": taverngym.N_CARDS,
        "n_actions": taverngym.N_ACTIONS,
    }


def save_checkpoint(path, model, optimizer, update, steps, config, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "engine": engine_signature(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "steps": steps,
        "env_config": config,
        "seed": seed,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["engine"] != engine_signature():
        raise ValueError(
            "Checkpoint engine or observation layout differs from this TavernGym build")
    model = Policy().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def require_ready(obs):
    if not np.all(obs["ready"]):
        raise RuntimeError("This collector supports one controlled seat per game, all ready")


def completed_placements(dones, info):
    if np.any(info["rejected"]):
        raise RuntimeError("The simulator rejected a masked policy action")
    places = info["placement"][dones]
    if np.any((places < 1) | (places > 8)):
        raise RuntimeError("The simulator stopped an episode without a valid placement")
    return places
