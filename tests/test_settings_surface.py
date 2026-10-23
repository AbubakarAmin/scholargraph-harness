"""Dashboard settings surface: every setting has help text and resolves to a live value.

Owner requirement: each parameter shows a simple title in Settings, with a
hover tooltip explaining what it does and how to set it. This pins:
(a) every settingDefs entry carries non-empty help,
(b) every non-secret id resolves via /api/config,
(c) every secret id resolves via /api/keys,
(d) every id is savable through apply_runtime_keys (upper-cased).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMIN_JS = (ROOT / "web" / "static" / "admin" / "admin.js").read_text(encoding="utf-8")
CONFIG_SRC = (ROOT / "core" / "config.py").read_text(encoding="utf-8")


def _defs():
    m = re.search(r"const settingDefs=\[(.*?)\];", ADMIN_JS, re.S)
    assert m, "settingDefs not found in admin.js"
    return re.findall(r"\['([^']+)','([^']+)'", m.group(1))


def _entries():
    m = re.search(r"const settingDefs=\[(.*?)\];", ADMIN_JS, re.S)
    return re.findall(r"\[[^\]]+\]", m.group(1))


def test_every_setting_has_help_text():
    entries = _entries()
    assert len(entries) >= 80, f"settings surface shrank: {len(entries)}"
    short = [e[:60] for e in entries if e.count(",") < 4]
    assert not short, f"settings without help text: {short}"


def test_every_setting_resolves_to_live_value():
    from web.app import get_config, get_keys

    values = get_config()
    keys = get_keys()["keys"]
    missing = [i for i, _ in _defs() if i != i.upper() and i not in values]
    assert not missing, f"settings with no /api/config value: {missing}"
    missing_keys = [i for i, _ in _defs() if i == i.upper() and i not in keys]
    # Unset optional secrets are correctly absent; anything else must resolve.
    assert set(missing_keys) <= {"SEMANTIC_SCHOLAR_API_KEY"}, missing_keys


def test_every_setting_is_savable():
    tree = ast.parse(CONFIG_SRC)
    mapping = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [c.value for c in node.keys if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if "LLM_PROVIDER" in keys and "RESEARCH_DOMAIN" in keys:
                mapping.update(keys)
    unmapped = [i for i, _ in _defs() if i.upper() not in mapping]
    assert not unmapped, f"settings the save path drops: {unmapped}"
