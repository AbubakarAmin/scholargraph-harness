"""Offline tests for independent reproducible execution."""

import json

import pytest

from agents.execution import ExecutionAgent


def test_execution_agent_records_seeded_results_and_hash(tmp_path):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path
    code = 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))'

    artifact = agent.execute(
        {"experiment_name": "toy", "source_code": code},
        {"experiment_name": "toy", "seeds": [7, 8]},
    )

    assert artifact["status"] == "completed"
    assert artifact["environment"]["seeds"] == [7, 8]
    assert artifact["content_hash"]
    assert json.loads(open(artifact["raw_results_path"], encoding="utf-8").read())["n_seeds"] == 2


def test_execution_agent_rejects_forbidden_code(tmp_path):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path

    artifact = agent.execute(
        {"experiment_name": "bad", "source_code": "import subprocess\nsubprocess.run([])"},
        {"experiment_name": "bad", "seeds": [1]},
    )

    assert artifact["status"] == "failed"
    assert "rejected" in artifact["error"].lower()


def test_arena_code_requires_docker_for_independent_validation(tmp_path, monkeypatch):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path
    code = 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))'
    calls = []

    monkeypatch.setattr(
        "core.container_sandbox.ensure_sandbox_image",
        lambda auto_build=True: {"ok": True, "backend": "docker"},
    )

    def _docker_run(code, **kwargs):
        calls.append(kwargs)
        seeds = [kwargs["base_seed"] + index * 1009 for index in range(kwargs["n_seeds"])]
        return {
            "success": True,
            "n_seeds": len(seeds),
            "n_success": len(seeds),
            "runs": [
                {"success": True, "seed": seed, "parsed": {"metrics": {"accuracy": 0.8}}}
                for seed in seeds
            ],
            "aggregate_metrics": {"accuracy": {"mean": 0.8, "std": 0.0, "n": len(seeds)}},
        }

    monkeypatch.setattr("core.container_sandbox.container_multi_seed", _docker_run)
    monkeypatch.setattr(
        "agents.execution.execute_multi_seed",
        lambda *args, **kwargs: pytest.fail("arena code must not use the configured dispatcher"),
    )
    artifact = agent.execute(
        {"experiment_name": "arena-toy", "source_code": code, "execution_backend": "arena"},
        {"experiment_name": "arena-toy", "seeds": [42, 1051]},
    )
    assert artifact["status"] == "completed"
    assert artifact["environment"]["sandbox"] == "Docker (arena-required)"
    assert calls[0]["base_seed"] == 42


def test_arena_code_does_not_fall_back_when_docker_is_unavailable(tmp_path, monkeypatch):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path
    code = 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))'

    monkeypatch.setattr(
        "core.container_sandbox.ensure_sandbox_image",
        lambda auto_build=True: {"ok": False, "reason": "docker_unreachable"},
    )
    monkeypatch.setattr(
        "agents.execution.execute_multi_seed",
        lambda *args, **kwargs: pytest.fail("arena code must not use AST fallback"),
    )
    artifact = agent.execute(
        {"experiment_name": "arena-toy", "source_code": code, "execution_backend": "arena"},
        {"experiment_name": "arena-toy", "seeds": [42]},
    )
    assert artifact["status"] == "failed"
    assert "Docker execution is required" in artifact["error"]


def test_arena_independent_validation_rejects_nonfinite_metrics(tmp_path, monkeypatch):
    agent = ExecutionAgent.__new__(ExecutionAgent)
    agent.output_dir = tmp_path
    code = 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))'
    monkeypatch.setattr(
        "core.container_sandbox.ensure_sandbox_image",
        lambda auto_build=True: {"ok": True, "backend": "docker"},
    )
    monkeypatch.setattr(
        "core.container_sandbox.container_multi_seed",
        lambda code, **kwargs: {
            "success": True,
            "n_seeds": 1,
            "n_success": 1,
            "runs": [{
                "success": True, "seed": 42,
                "parsed": {"metrics": {"accuracy": float("nan")}},
            }],
            "aggregate_metrics": {"accuracy": {"mean": float("nan"), "std": 0.0, "n": 1}},
        },
    )
    artifact = agent.execute(
        {"experiment_name": "arena-toy", "source_code": code, "execution_backend": "arena"},
        {"experiment_name": "arena-toy", "seeds": [42]},
    )
    assert artifact["status"] == "failed"
    assert "not finite" in artifact["error"]