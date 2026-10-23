# ScholarGraph runtime skills

These skills are runtime guidance for ScholarGraph's prompt-registry agents.
They are separate from VS Code's `.github/skills/` discovery location.

Each skill uses the portable Agent Skills `SKILL.md` shape: a matching `name`,
a keyword-rich `description`, and a concise procedure. Prompt-level routing is
explicit in `core/prompts.py`; only the selected prompt gets a skill loaded.
Long research evidence remains in the call's supplied artifacts, never in a
persistent skill.

| Skill | Main routed work |
|---|---|
| `evidence-synthesis` | Topic discovery/ranking, thesis construction, review screening and extraction |
| `debate-rubric` | Proposer, challenger, follow-up, moderator, and debate repair |
| `research-design` | Debate and experiment planning; falsifiability |
| `sandbox-fix-hints` | Engineer repair hints for known failure categories |
| `experimental-reproducibility` | Engineer generation, refinement, and ablation |
| `writer-number-policy` | All manuscript sections; exact numbers, citations, and claim scope |
| `statistical-interpretation` | Quantitative writing and supervision |
| `peer-review` | Supervisor review and systematic-review tasks |
| `operator-handoff` | Meta Agent run questions and performance recommendations |

The loader checks `skills/<name>/SKILL.md` first and retains compatibility
with old flat `skills/<name>.md` files. `validate_skill_registry()` checks
frontmatter, prompt routes, and non-empty bodies. `render_system()` injects
only routed skills within the existing prompt-system budget. Skill content is
cached for consistent use; restart the application after changing skill files.
Prompt/run state records each used skill's SHA-256 under
`prompt_version["skill:<name>"]`.

Skills are instructions, not permissions. Tool permissions and execution
boundaries remain enforced by ScholarGraph's capability broker and sandbox.
The Data, Execution, Analysis, and Verification agents primarily perform
deterministic work; their correctness is enforced by contracts and tests rather
than by adding unused LLM prompt text.

The research behind these design choices is documented in
[Agent Skills: Research and ScholarGraph Design](../docs/agent-skills.md).

Run the focused checks with:

```powershell
python -m pytest tests/test_agentic_upgrade.py -q
```

Design references:

- [Agent Skills specification](https://agentskills.io/specification)
- [OpenClaw skills and gating](https://docs.openclaw.ai/tools/skills)
- [Hermes Agent progressive skill loading](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills#progressive-disclosure)
- [OpenCode skills](https://opencode.ai/docs/skills/)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
