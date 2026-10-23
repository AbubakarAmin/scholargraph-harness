"""Generate a report for the latest failed and completed runs."""

from __future__ import annotations

import argparse

from core.forensics import write_historical_report


parser = argparse.ArgumentParser()
parser.add_argument("output_path")
args = parser.parse_args()
print(write_historical_report(args.output_path))