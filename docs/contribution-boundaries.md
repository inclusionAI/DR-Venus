# Contribution boundaries

## Upstream work

| Layer | Attribution | What may be said |
|---|---|---|
| DR-Venus model and official scores | inclusionAI / Venus Team | “Based on and reproduced from the open-source DR-Venus project.” |
| IGPO algorithm | Wang et al., ICLR 2026 | “Implemented/validated an IGPO ablation”; never “invented IGPO.” |
| Distributed RL infrastructure | verl | “Configured and debugged verl/FSDP/vLLM”; never count vendored framework lines as personal code. |
| Agent scaffold and tool protocol | DR-Venus, DeepResearcher, Tongyi DeepResearch, REDSearcher | Attribute the inherited loop, prompts, and data sources. |

## Personal reproduction work

The personal scope is limited to changes and evidence present in this fork:

- correctness filtering and auditable question-level data splits;
- equal-assistant-token SFT sampling arms;
- online/record/replay tool responses;
- fixed evaluation subsets, manifests, raw predictions, confidence intervals,
  cost accounting, and failure analysis;
- budget-matched 4B SFT and short-horizon GRPO/IGPO/format ablations;
- reproducibility, safety, tests, and interview documentation.

## Prohibited claims

- Do not write “trained DR-Venus-4B to BrowseComp 29.1” without independently
  reproducing that exact official setting.
- Do not compare a 32K/12-turn personal score directly against a 256K/200-turn
  official score as though budgets were equal.
- Do not report the better seed while hiding failed runs.
- Do not imply that cached web responses equal a live-web benchmark.

Every resume number must link to an experiment manifest, immutable prediction
file, evaluation report, and commit.
