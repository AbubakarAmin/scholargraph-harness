"""Independent deterministic verification of research artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from core.capabilities import DEFAULT_MANIFESTS
from core.context import RunContext, get_active_context
from core.contracts import ExecutionArtifact, StatisticalReport, VerificationFinding
from core.verification import verify_statistics

logger = logging.getLogger(__name__)


class VerificationAgent:
    """Verify outputs produced by other workers without editing their artifacts."""

    capability_manifest = DEFAULT_MANIFESTS["VerificationAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    def verify(
        self,
        execution_artifacts: Mapping[str, ExecutionArtifact],
        analysis_reports: Mapping[str, StatisticalReport],
    ) -> list[VerificationFinding]:
        findings: list[VerificationFinding] = []
        for name, execution in execution_artifacts.items():
            findings.extend(self._verify_execution(name, execution, analysis_reports.get(name)))
        return findings

    def _verify_execution(
        self,
        name: str,
        execution: ExecutionArtifact,
        report: Optional[StatisticalReport],
    ) -> list[VerificationFinding]:
        findings: list[VerificationFinding] = []
        path = Path(execution.get("raw_results_path", ""))
        if not path.is_file():
            return [self._finding(name, "missing_raw_results", "Raw execution results are missing", True)]
        raw_bytes = path.read_bytes()
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        if actual_hash != execution.get("content_hash"):
            findings.append(self._finding(name, "content_hash", "Raw result hash does not match execution artifact", True))
        if execution.get("status") != "completed":
            findings.append(self._finding(name, "execution_status", "Execution did not complete", True))
        if report:
            result = verify_statistics(report, raw_results_path=str(path))
            if not result.get("passed"):
                findings.append(self._finding(name, "statistics", result.get("note", "Statistical report disagrees with raw results"), True, result))
        else:
            findings.append(self._finding(name, "missing_analysis", "No independent statistical report supplied", True))
        return findings

    @staticmethod
    def _finding(
        experiment: str,
        check: str,
        message: str,
        blocking: bool,
        evidence: Optional[dict[str, Any]] = None,
    ) -> VerificationFinding:
        return {
            "finding_id": f"{experiment}:{check}",
            "severity": "error" if blocking else "warning",
            "check": check,
            "message": message,
            "artifact_ids": [experiment],
            "blocking": blocking,
            "status": "failed" if blocking else "passed",
            "evidence": evidence or {},
        }