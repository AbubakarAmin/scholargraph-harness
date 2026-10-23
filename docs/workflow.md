# Research workflow

The production graph is recipe-aware: intake selects a paper type, and the graph advances through that type's configured phases. Experimental work is the most complete path, but it is not the only workflow.

## Supported paper types

The current registry in `core/phase_registry.py` defines:

- Experimental, comparative, and empirical
- Review
- Analytical and argumentative
- Descriptive and survey
- Cause-effect
- Case study

Each type has an ordered recipe and evidence requirements. The recipe may skip phases that are not appropriate—for example, a review can use systematic-review and thesis phases instead of experiment engineering. The exact recipe definitions in `core/phase_registry.py` are authoritative.

## Typical experimental path

```text
intake
  → topic discovery
  → hypothesis debate
  → planning
  → data validation
  → narrative writing
  → engineering
  → independent validation
  → results writing
  → supervision
  → editing
```

Other recipes can add systematic review, thesis argument, survey analysis, case protocol, or causal audit phases, and may omit debate, engineering, or independent validation. The graph's runtime recipe determines which phases execute.

## Evidence and review

- Discovery retrieves and organizes research evidence. Candidate topics are screened before proceeding.
- Debate and planning refine questions, hypotheses, methods, and required evidence where those phases are part of the recipe.
- Data validation checks declared inputs for applicable workflows.
- Experiment contracts and evidence gates constrain engineering runs where experiments are required.
- Independent validation can replay or analyze experiment results separately from their generation.
- Supervision and editing apply quality checks and assemble the output. Passing checks means the result is reviewable, not that it is scientifically proven or publication-ready.

## Routing and recovery

- The workflow may retry a bounded repair or revision path where configured; it does not treat every error as recoverable.
- A terminal error or exhausted budget stops further phases. Partial run state and artifacts may remain available for diagnosis.
- A plan revision requested during engineering routes back to planning when supported by the active graph.
- Editing may route a bounded repair back to results writing when the editor reports repairable findings.
- Durable checkpoints support resuming a run with `python main.py --resume RUN_ID`.

The graph implementation lives in `core/workflow.py`; individual node behavior is implemented in `core/workflow_nodes.py`.
