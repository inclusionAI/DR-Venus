#!/usr/bin/env python3
"""Run the public, CPU-only reproducibility smoke suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run("-m", "pytest", "-q")
    run("-m", "evaluation.build_eval_core", "--help")
    run("-m", "evaluation.evaluate", "--help")
    run(str(ROOT / "experiments" / "prepare_rl_subset.py"), "--help")
    run(str(ROOT / "SFT" / "data_clean" / "prepare_trajectories.py"), "--help")
    run(str(ROOT / "SFT" / "data_clean" / "build_sft_arms.py"), "--help")
    run(str(ROOT / "experiments" / "budget_guard.py"), "--help")
    run(str(ROOT / "experiments" / "check_budget_report.py"), "--help")
    run(str(ROOT / "experiments" / "verify_protocol_data.py"))
    run(str(ROOT / "scripts" / "check_no_secrets.py"))
    print("CPU smoke passed. GPU training and live API calls were not exercised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
