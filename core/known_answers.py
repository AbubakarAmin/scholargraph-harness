"""Known-answer probes for common deterministic experiment primitives."""

from __future__ import annotations

from typing import Any, Dict, Optional


FIXTURES: Dict[str, Dict[str, Any]] = {
    "identity_transform": {
        "metrics": {"identity_error": 0.0},
        "code": 'import json\nprint(json.dumps({"metrics": {"identity_error": 0.0}}))',
        "tolerance": 1e-6,
    },
    "linear_eigenvalue": {
        "metrics": {"largest_eigenvalue": 2.0},
        "code": 'import json\nprint(json.dumps({"metrics": {"largest_eigenvalue": 2.0}}))',
        "tolerance": 1e-6,
    },
}


def fixture_for(experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = experiment.get("known_answer_type")
    return dict(FIXTURES[name]) if name in FIXTURES else None
