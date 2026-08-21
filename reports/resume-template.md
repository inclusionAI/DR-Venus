# Resume and interview claim template

Do not fill numeric placeholders until the corresponding artifacts exist.

- Reproduced and extended the open-source DR-Venus 4B deep-research agent,
  adding auditable correctness filtering, question-level data isolation, and
  record/replay web tools; produced deterministic data/model/experiment
  manifests and a fixed `N=<...>` evaluation set.
- Under a matched `<context>/<max-turns>/<sample-budget>` setup, compared
  uniform versus long-trajectory SFT at equal assistant-token budget and
  measured `<metric: X -> Y, 95% CI ...>` using `<GPU-hours/cost>`.
- Implemented a controlled outcome-only GRPO / IGPO / turn-format ablation over
  two seeds; diagnosed `<finding>` through outcome diversity, information-gain
  activation, KL, browse ratio, and `<count>` manually classified failures.

Mandatory qualifier: this is a budget-matched personal reproduction based on
DR-Venus/IGPO/verl, not a reproduction of the official 200-turn score.
