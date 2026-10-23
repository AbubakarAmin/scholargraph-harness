# Agent Skills: Research and ScholarGraph Design

## What the other systems do well

The reviewed systems share a useful skills convention: a concise `SKILL.md`
with a required `name` and task-matching `description`, then optional references
and scripts loaded only when a task needs them. They differ mainly in how
skills are routed, gated, and maintained.

- **OpenClaw** separates skill discovery, runtime readiness, and per-agent
  visibility. It supports requirements-based gating and managed skill
  revisions. Its documentation explicitly warns that visibility allowlists
  are not shell security boundaries.
- **Hermes Agent** documents progressive disclosure: a compact skill index,
  the full skill instructions, then selected reference files. Its curator also
  tracks usage and stale skills, while named bots can have separate role,
  memory, and skill settings.
- **OpenCode** combines on-demand skills with role-specialized agents,
  descriptions for routing, and per-agent tool permissions.
- **Claude Code** supports skill invocation controls and separate subagent
  contexts; subagents can have their own tools and selected skills.
- **Agent Skills specification** provides portable metadata and the
  `SKILL.md` layout, but notes that tool grants are experimental and
  implementation-dependent.

## Decisions for ScholarGraph

1. Keep each agent's identity, schema, and hard gates in the versioned prompt
   registry. Skills add reusable procedures; they do not replace those roles.
2. Route by prompt ID, not by broad agent name. A results-writing call receives
   statistical interpretation; a debate moderator receives the debate rubric.
3. Load instructions only when the routed prompt is rendered. Keep paper
   evidence and run-specific context in the supplied artifacts, not in shared
   skill files.
4. Validate skill metadata and routing in offline tests. Use the compatible
   standard `SKILL.md` shape while keeping legacy flat files loadable.
5. Hash the exact loaded skill file and record it with prompt versions in run
   state, so changed procedures are visible in reproducibility metadata.
6. Keep enforcement in Python. Skills do not grant tools, authorize actions,
   or turn the AST sandbox into an OS security boundary.
7. Do not add prompt guidance to deterministic-only agents merely to increase
   the skill count. Their quality belongs in typed contracts, implementation,
   and executable tests.

## Current runtime map

| Work | Skill guidance |
|---|---|
| Discovery, thesis, evidence extraction | `evidence-synthesis` |
| Hypothesis debate | `debate-rubric`, `research-design` |
| Experiment planning | `research-design` |
| Engineering | `sandbox-fix-hints`, `experimental-reproducibility` |
| Manuscript writing | `writer-number-policy`; quantitative sections also receive `statistical-interpretation` |
| Review and supervision | `peer-review`; statistical review also receives `statistical-interpretation` |
| Meta Agent | `operator-handoff` |

The live route table and frontmatter checks are in `core/prompts.py`. The
focused regression test is `tests/test_agentic_upgrade.py`.

## Sources

- [Agent Skills specification](https://agentskills.io/specification)
- [OpenClaw skills: loading order and agent allowlists](https://docs.openclaw.ai/tools/skills#loading-order)
- [OpenClaw skills: gating](https://docs.openclaw.ai/tools/skills#gating)
- [OpenClaw Skill Workshop: undo](https://docs.openclaw.ai/tools/skill-workshop#undo)
- [Hermes Agent skills: progressive disclosure](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills#progressive-disclosure)
- [Hermes Agent skills: conditional activation](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills#conditional-activation-fallback-skills)
- [Hermes Agent Curator](https://hermes-agent.nousresearch.com/docs/user-guide/features/curator)
- [OpenCode skills](https://opencode.ai/docs/skills/)
- [OpenCode agents and permissions](https://opencode.ai/docs/agents/)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Claude Code subagents](https://code.claude.com/docs/en/sub-agents)
