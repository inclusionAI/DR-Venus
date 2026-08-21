# Core glossary and mastery targets

| Concept | Working definition | Required mastery |
|---|---|---|
| Agentic SFT | Teacher-forced next-token learning over reasoning/action spans while environment observations are masked | L4: mark the loss mask by hand and debug token alignment |
| GRPO | Group-relative policy optimization using several rollouts for a shared prompt and a group baseline | L4: derive the clipped objective and explain advantage collapse |
| IGPO | Dense turn reward from the change in policy log-probability of the ground truth across adjacent histories | L4: hand-compute IG, normalization, discount, and token broadcast |
| Outcome reward | Terminal correctness signal from F1/rules or an LLM judge | L4: discuss judge noise, hacking, and reproducibility |
| Turn-level format penalty | A fixed non-positive value that **replaces** the malformed turn's IG/outcome reward (it is not additive) | L4: explain replacement semantics and its effect on R2 |
| FSDP | Parameter, gradient, and optimizer-state sharding across data-parallel workers | L3: draw placement and debug common OOM/NCCL failures |
| Tensor parallelism | Split model tensor operations across devices, used by vLLM rollout here | L3: distinguish it from DP and SP |
| Ulysses sequence parallelism | Split a long sequence across devices for attention/training | L3: explain communication and divisibility constraints |
| KV cache | Stored attention keys/values that avoid recomputing prefixes during generation or IG scoring | L3: estimate memory growth and explain cache invalidation |
| Record/replay tools | Capture immutable external-tool responses once and reuse them for controlled comparisons | L4: explain fairness benefits and residual snapshot bias |
| Pass@K | Probability/empirical rate that at least one of K sampled attempts succeeds | L4: compute and avoid mixing it with Pass@1 |
| Paired bootstrap | Resample paired per-question outcomes to estimate an uncertainty interval for model differences | L4: interpret a CI crossing zero |
