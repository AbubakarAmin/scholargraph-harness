"""Replay a generated companion repository.

Usage: python replay_run.py OUTPUT/companion_repo
"""

from __future__ import annotations

import argparse
import json

from core.replay import replay_companion


parser = argparse.ArgumentParser()
parser.add_argument("companion_dir")
parser.add_argument("--clean-env", action="store_true")
args = parser.parse_args()
print(json.dumps(replay_companion(args.companion_dir, clean_env=args.clean_env), indent=2, default=str))
