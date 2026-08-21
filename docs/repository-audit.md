# Repository audit

Audit date: 2026-08-14.

## Confirmed gaps in the upstream checkout

1. The checkout initially matched its upstream branch and contained no personal
   implementation; only local IDE metadata was untracked.
2. The paper and root README describe correctness filtering from 10,001 to
   9,365 SFT trajectories, but the original cleaner did not contain a final
   answer versus ground-truth judge path.
3. The paper describes roughly 1K RL questions, while the checked-in
   `RL/data/train.parquet` contains 80,000 rows. No committed construction
   manifest explains the difference.
4. The original SFT launcher defaulted `VAL_FILES` to `TRAIN_FILES`, making the
   default validation number unsuitable as held-out evidence.
5. The repository did not include the benchmark harness, raw predictions,
   training logs, configuration snapshots, or hashes needed to reproduce the
   official result table.
6. Dependencies are not fully locked across CUDA, torch, flash-attn, and vLLM.
7. A tracked `.env`, Python bytecode, and OS metadata created avoidable secret
   and repository-hygiene risk.

## Questions that must be resolved before a paid run

- Which exact correctness-filtered SFT artifact corresponds to the paper?
- Which 1K RL subset and selection rule correspond to the official run?
- Which upstream commits of verl, IGPO, and DeepResearcher were used?
- Which judge and summarizer checkpoint/API versions produced the official
  rewards and tool observations?
- What were the official wall time, GPU-hours, and external API volume?

Until those questions have primary evidence, the personal experiments use
their own frozen manifests and are reported as budget-matched ablations.
