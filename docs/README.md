# DR-Venus personal reproduction track

This directory separates the personal, evidence-backed reproduction work from
the upstream DR-Venus implementation. It does not claim authorship of the
official model, IGPO, or verl.

## Start here

- [12-week learning and execution plan](learning-plan.md)
- [End-to-end architecture](architecture.md)
- [Contribution boundaries](contribution-boundaries.md)
- [Repository audit and open questions](repository-audit.md)
- [Core glossary](glossary.md)
- [Progress and weekly gates](progress.md)
- [Interview question bank](interview-question-bank.md)
- [Credential safety and publication gate](security.md)

Executable reproduction assets live in:

- `SFT/data_clean/`: correctness filtering, deterministic splits, manifests,
  and equal-token SFT arms.
- `evaluation/`: fixed evaluation-set construction and evidence-backed metrics.
- `experiments/`: dry-run-first SFT/RL profiles and budget guard.
- `drvenus_repro/`: shared cache and manifest utilities.

No official score is a personal result. A result becomes resume-eligible only
when its manifest, raw predictions, logs, model/data hashes, and evaluation
budget are present in this fork.
