---
name: sandbox-fix-hints
description: "Use when repairing experiment code failures involving timeouts, imports, sandbox restrictions, or output JSON."
metadata:
  scholargraph-stage: engineering
  scholargraph-roles: "EngineerAgent"
---

# Sandbox Fix Hints

## timeout
The previous run exceeded its time budget. Shrink the workload (n_samples <= 200, fewer iterations); vectorize and avoid heavy plots.

## import
Only these imports are allowed: numpy, pandas, matplotlib, sklearn, scipy, math, statistics, random, json, re, collections, seaborn, networkx, sympy. You MUST explicitly import every symbol you use; bare names without an import will raise NameError even when the root package is allowed. Replace other imports.

## sandbox
No file writes, subprocesses, sockets, exit, os/sys/requests/pathlib, eval, or exec. Compute in memory.

## json
Print exactly one final JSON line: {"metrics": {...}, "raw": {...}}. No other stdout or markdown.
