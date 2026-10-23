---
name: operator-handoff
description: "Use when answering an operator about a live research run, assessing system performance, or recommending the next run action."
metadata:
  scholargraph-stage: orchestration
  scholargraph-roles: "MetaAgent"
---

# Operator Handoff

## Procedure
1. Use only current run summaries, events, metrics, and supplied history; separate facts from interpretation.
2. Tie the smallest concrete next action to a named failure, metric, or phase.
3. If evidence is missing, say so or request clarification. Never imply a suggested control was applied.
4. Use only allowed control values. Escalate scope or evidence problems; do not silently change thresholds or results.

## Output Check
Recommendations are grounded in run data; controls and keys match the schema; no credentials or unrelated private history are exposed.
