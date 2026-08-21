#!/usr/bin/env python3
"""Fail without echoing values when credential-like material is detected."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_NAMES = {
    "API_KEY",
    "JUDGE_API_KEY",
    "SERPER_KEY_ID",
    "JINA_API_KEYS",
    "OPENAI_API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "CLIENT_SECRET",
    "PASSWORD",
}
ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
TOKEN_SHAPE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[oprsu]_[A-Za-z0-9]{20,})\b")
PLACEHOLDERS = (
    "your-",
    "example",
    "placeholder",
    "changeme",
    "replace-me",
    "test-only",
    "dummy",
)
SKIP_DIRS = {".git", ".idea", ".venv", "__pycache__", "artifacts", "outputs", "output"}
TEXT_SUFFIXES = {
    "",
    ".env",
    ".example",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").casefold()
    return (
        not normalized
        or normalized.startswith("${")
        or normalized.startswith("<")
        or "os.getenv(" in normalized
        or "os.environ.get(" in normalized
        or any(marker in normalized for marker in PLACEHOLDERS)
    )


def scan_text(text: str, *, location: str) -> list[str]:
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = ASSIGNMENT.match(line)
        if match and match.group(1) in SENSITIVE_NAMES and not _placeholder(match.group(2)):
            findings.append(f"{location}:{line_number}: non-placeholder {match.group(1)} assignment")
        if PRIVATE_KEY.search(line):
            findings.append(f"{location}:{line_number}: private-key header")
        if TOKEN_SHAPE.search(line):
            findings.append(f"{location}:{line_number}: credential-shaped token")
    return findings


def scan_worktree(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES and path.name not in {".env", ".env.example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings.extend(scan_text(text, location=path.relative_to(root).as_posix()))
        if path.name == ".env":
            findings.append(f"{path.relative_to(root).as_posix()}: populated .env file present")
    return findings


def scan_env_history(root: Path = ROOT) -> list[str]:
    """Inspect historical .env patches while never returning secret values."""

    completed = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "--format=commit:%H",
            "-p",
            "--",
            ":(glob)**/.env",
            ":(exclude,glob)**/.env.example",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    findings: list[str] = []
    commit = "unknown"
    file_path = ".env"
    for line in completed.stdout.splitlines():
        if line.startswith("commit:"):
            commit = line.split(":", 1)[1][:12]
        elif line.startswith("+++ b/"):
            file_path = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            for finding in scan_text(line[1:], location=f"history:{commit}:{file_path}"):
                findings.append(finding.rsplit(":", 2)[0] + ": credential material added")
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also inspect historical .env patches (required before publication)",
    )
    args = parser.parse_args(argv)
    findings = scan_worktree()
    if args.history:
        findings.extend(scan_env_history())
    if findings:
        print(f"Secret scan failed with {len(findings)} finding(s); values are intentionally redacted:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Secret scan passed; no credential-like values were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
