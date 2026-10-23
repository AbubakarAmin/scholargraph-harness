"""Generate a durable claim-lineage report.

Usage: python forensic_report.py RUN_ID OUTPUT/report.json
"""

from __future__ import annotations

import argparse

from core.forensics import write_forensic_report


parser = argparse.ArgumentParser()
parser.add_argument("run_id")
parser.add_argument("output_path")
args = parser.parse_args()
print(write_forensic_report(args.run_id, args.output_path))
