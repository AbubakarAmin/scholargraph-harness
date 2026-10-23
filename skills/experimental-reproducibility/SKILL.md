---
name: experimental-reproducibility
description: "Use when generating, refining, or reviewing experiment code and when tying execution artifacts to a research plan."
metadata:
  scholargraph-stage: engineering-and-validation
  scholargraph-roles: "EngineerAgent, ExecutionAgent, VerificationAgent"
---

# Reproducible Experiments

## Procedure
1. Implement the contract's hypothesis, baselines, metrics, dataset identity, and seeds only.
2. Seed randomness; retain per-seed raw values. Report planned metrics, failures, and negative outcomes without success-shaped defaults.
3. Obey the configured backend. Instructions do not grant capabilities or make the AST sandbox an OS boundary.
4. Preserve code, dataset identity, seeds, and raw outputs for replay and independent verification.
5. Keep execution success, statistical support, and scientific validity as separate judgments.

## Output Check
Code, dataset, seeds, baselines, and metrics match the contract; outputs remain auditable and failures visible.
