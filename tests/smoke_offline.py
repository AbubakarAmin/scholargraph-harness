"""Quick smoke test for research-grade upgrades (no API keys required)."""
import sys
from pathlib import Path

# Support the documented direct invocation: ``python tests/smoke_offline.py``.
# Python otherwise places only ``tests/`` on sys.path, hiding the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sandbox import validate_code, execute_sandboxed, run_multi_seed
from agents.planner import PlannerAgent
from agents.hypothesis_debate import DebateResult
from agents.engineer import EngineerAgent
from core.run_log import CrossRunMemory
from core.verification import extract_citation_ids, verify_statistics
import tempfile

ok, err = validate_code("import subprocess\nsubprocess.run(['x'])")
assert not ok, err
print("PASS sandbox blocks subprocess:", err)

ok2, err2 = validate_code("exit()")
assert not ok2
print("PASS sandbox blocks exit")

r = execute_sandboxed('import json\nprint(json.dumps({"metrics": {"a": 1.0}}))')
assert r["success"], r
assert r["parsed"]["metrics"]["a"] == 1.0
print("PASS sandboxed exec")

m = run_multi_seed(
    'import json, random\nprint(json.dumps({"metrics": {"acc": random.random()}}))',
    n_seeds=3,
)
assert m["success"] and "acc" in m["aggregate_metrics"]
print("PASS multi-seed", m["aggregate_metrics"]["acc"])

p = PlannerAgent.__new__(PlannerAgent)
assert p._flag_unfalsifiable({"expected_contributions": ["x"]})
assert p._flag_missing_baselines({"experiments": [{"name": "M", "baselines": []}]})
print("PASS planner flags")

assert "rounds" in DebateResult.__dataclass_fields__
print("PASS debate fields")

eng = EngineerAgent.__new__(EngineerAgent)
cc = eng.check_code_claim_consistency(
    {"baselines": ["gradient_boosting", "xgboost"], "claimed_components": []},
    "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()",
)
assert cc["score"] < 8 or not cc["consistent"]
print("PASS code-claim check", cc)

ids = extract_citation_ids("see doi:10.1038/nature14539 and arXiv:1706.03762")
assert ids["dois"] and ids["arxiv_ids"]
print("PASS citation extract", ids)

raw = {"aggregate_metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}
assert verify_statistics({"metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}, raw_data=raw)["passed"]
assert not verify_statistics({"metrics": {"accuracy": {"mean": 0.99}}}, raw_data=raw)["passed"]
print("PASS stats verify")

with tempfile.TemporaryDirectory() as d:
    mem = CrossRunMemory(path=str(Path(d) / "c.jsonl"))
    mem.record_rejection("topic", "foo", "novelty")
    assert "foo" in mem.lessons_for_prompt()
print("PASS cross-run memory")

print("\nALL SMOKE TESTS PASSED")
