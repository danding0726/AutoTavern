"""PPO correctness checks; skipped when the optional torch dependency is absent."""

import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

import taverngym
from taverngym.evaluate import evaluate
from taverngym.policy import (
    Policy,
    completed_placements,
    load_checkpoint,
    observation_tensors,
    require_ready,
    save_checkpoint,
)
from taverngym.train import collect_rollout, compute_gae, main, ppo_update


@pytest.fixture(autouse=True)
def torch_setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def test_masked_policy_never_selects_illegal_actions():
    model = Policy()
    obs = taverngym.BatchEnv(2, seed=7).observe()
    # Include a forced-action row to exercise entropy with almost every logit masked.
    obs["action_mask"][0] = False
    obs["action_mask"][0, 30] = True
    distribution, values = model(*observation_tensors(obs, "cpu"))
    mask = torch.as_tensor(obs["action_mask"])
    assert torch.all(distribution.probs[~mask] == 0)
    assert torch.isfinite(distribution.entropy()).all()
    assert torch.isfinite(values).all()
    actions = distribution.sample((100,))
    assert mask[torch.arange(2), actions].all()
    obs["action_mask"][1] = False
    with pytest.raises(ValueError, match="legal action"):
        model(*observation_tensors(obs, "cpu"))


def test_gae_cuts_resets_and_bootstraps_rollout_boundary():
    rewards = torch.tensor([[0.0], [1.0], [0.0]])
    dones = torch.tensor([[False], [True], [False]])
    values = torch.tensor([[0.2], [0.3], [10.0]])
    advantages, returns = compute_gae(rewards, dones, values, torch.tensor([20.0]), 0.9, 0.8)
    # t=1 is terminal: neither the new episode's value 10 nor its advantage may leak back.
    expected = torch.tensor([[0.07 + 0.9 * 0.8 * 0.7], [0.7], [8.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


def test_real_rollout_update_and_checkpoint(tmp_path):
    config = {"turn_action_caps": [4]}
    env = taverngym.BatchEnv(2, seed=3, controlled=(0,), config=config, threads=2)
    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    initial = model.actor.weight.detach().clone()
    rollout, _, next_value, placements = collect_rollout(env, model, env.observe(), 128, "cpu")
    assert placements and all(1 <= p <= 8 for p in placements)
    assert rollout["dones"].any()
    assert rollout["action_mask"].gather(-1, rollout["actions"].unsqueeze(-1)).all()
    expected_rewards = 1 - 2 * (np.array(placements) - 1) / 7
    np.testing.assert_allclose(rollout["rewards"][rollout["dones"]].numpy(), expected_rewards,
                               atol=1e-6)
    advantages, returns = compute_gae(rollout["rewards"], rollout["dones"], rollout["values"],
                                       next_value, 0.999, 0.99)
    metrics = ppo_update(model, optimizer, rollout, advantages, returns, 2, 64)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(model.actor.weight, initial)
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, optimizer, 1, 256, config, 3)
    restored, checkpoint = load_checkpoint(path)
    assert checkpoint["update"] == 1 and checkpoint["optimizer"]["state"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    checkpoint["engine"]["source_hash"] = "other-build"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="engine"):
        load_checkpoint(path)


def test_training_can_resume_and_refuses_overwrite(tmp_path):
    args = ["--out", str(tmp_path), "--updates", "1", "--n-envs", "2",
            "--rollout-steps", "8", "--epochs", "1", "--minibatch-size", "16",
            "--device", "cpu", "--action-cap", "4"]
    first = main(args)
    second = main(args + ["--resume", str(tmp_path / "latest.pt")])
    assert (first["update"], first["steps"]) == (1, 16)
    assert (second["update"], second["steps"]) == (2, 32)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    with pytest.raises(SystemExit):
        main(args)


def test_evaluation_is_reproducible_and_counts_complete_games():
    config = {"turn_action_caps": [4]}
    first = evaluate(None, 3, 100_000, config, "cpu")
    second = evaluate(None, 3, 100_000, config, "cpu")
    assert first == second
    assert sum(first["placement_counts"].values()) == 3
    assert first["illegal_actions"] == 0
    assert 1 <= first["mean_placement"] <= 8
    model_result = evaluate(Policy(), 1, 100_000, config, "cpu")
    assert model_result["games"] == 1 and len(model_result["placements"]) == 1


def test_collector_rejects_waiting_slots_and_stopped_episodes():
    with pytest.raises(RuntimeError, match="ready"):
        require_ready({"ready": np.array([True, False])})
    with pytest.raises(RuntimeError, match="placement"):
        completed_placements(np.array([True]), {"rejected": np.array([False]),
                                                "placement": np.array([0])})
    with pytest.raises(RuntimeError, match="rejected"):
        completed_placements(np.array([False]), {"rejected": np.array([True]),
                                                 "placement": np.array([0])})
