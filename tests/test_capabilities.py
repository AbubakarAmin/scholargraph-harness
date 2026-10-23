"""Contract tests for scoped agent capabilities."""

import pytest

from core.capabilities import DEFAULT_MANIFESTS, authorize, manifest_for
from core.tool_broker import CapabilityBroker


def test_engineer_cannot_execute_or_analyze():
    engineer = DEFAULT_MANIFESTS["EngineerAgent"]

    assert authorize(engineer, "code.generate").allowed
    # Scoped development tokens (post-M6 review): sandbox pre-runs of own
    # candidate code + Tier-1 brokered dataset fetch — never validation.
    assert authorize(engineer, "code.execute_sandbox").allowed
    assert authorize(engineer, "dataset.download_brokered").allowed
    # Validation-grade counterparts stay forbidden: the engineer must never
    # self-validate (independent replay belongs to ExecutionAgent) nor pull
    # raw outbound data (belongs to DataAgent).
    assert not authorize(engineer, "code.execute").allowed
    assert not authorize(engineer, "dataset.download").allowed
    assert not authorize(engineer, "analysis.statistics").allowed


def test_verifier_can_read_and_replay_but_not_change_code():
    verifier = DEFAULT_MANIFESTS["VerificationAgent"]

    assert authorize(verifier, "artifact.read").allowed
    assert authorize(verifier, "verification.replay").allowed
    assert not authorize(verifier, "code.generate").allowed


def test_unknown_capabilities_fail_closed():
    decision = authorize(DEFAULT_MANIFESTS["EngineerAgent"], "network.browser")

    assert not decision.allowed
    assert decision.reason == "unknown capability"


def test_manifest_rejects_unknown_or_overlapping_capabilities():
    with pytest.raises(ValueError, match="Unknown capabilities"):
        manifest_for("bad", "test", {"network.browser"})

    with pytest.raises(ValueError, match="both allowed and forbidden"):
        manifest_for("bad", "test", {"artifact.read"}, {"artifact.read"})


def test_broker_authorizes_and_audits_tool_calls():
    broker = CapabilityBroker({"read_artifact": lambda name: f"read:{name}"})

    result = broker.call(
        DEFAULT_MANIFESTS["VerificationAgent"],
        "artifact.read",
        "read_artifact",
        name="results.json",
    )

    assert result == "read:results.json"
    assert broker.calls[-1].status == "started"


def test_broker_denies_before_dispatching():
    called = []
    broker = CapabilityBroker({"execute": lambda: called.append(True)})

    with pytest.raises(PermissionError):
        broker.call(DEFAULT_MANIFESTS["EngineerAgent"], "code.execute", "execute")

    assert called == []
    assert broker.calls[-1].status == "denied"


def test_gpu_checker_word_boundary_e0ec9e91():
    """e0ec9e91: 'output' must not fire 'tpu'; '"gpu": false' is not a requirement."""
    from core.capabilities import _gpu_required_by_text, check_plan_feasibility
    assert not _gpu_required_by_text("probabilistic and coherent output, no explicit post-processing")
    assert not _gpu_required_by_text('required_resources: {"gpu": false}')
    assert not _gpu_required_by_text("runs CPU-only and does not require a GPU")
    assert _gpu_required_by_text("train with distributed training on 8 GPUs")
    assert _gpu_required_by_text("requires CUDA kernel optimization")
    plan = {
        "title": "CPU study",
        "methodology": "Closed-form Gaussian output scoring on CPU",
        "experiments": [{"name": "mve", "dataset": {"name": "bundled_synthetic"}}],
    }
    assert check_plan_feasibility(plan) == []


def test_broker_permits_scoped_sandbox_but_not_replay():
    called = []
    broker = CapabilityBroker({"pre_run": lambda: called.append(True)})

    broker.call(DEFAULT_MANIFESTS["EngineerAgent"], "code.execute_sandbox", "pre_run")
    assert called == [True]

    with pytest.raises(PermissionError):
        broker.call(DEFAULT_MANIFESTS["EngineerAgent"], "code.execute", "pre_run")
