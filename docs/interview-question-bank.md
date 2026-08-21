# Interview question bank

Use these for three mock interviews. Each answer should cite a formula, code
location, or personal artifact rather than rely on slogans.

1. What problem does DR-Venus solve, and why is 4B important?
2. Which parts are upstream and which parts did you personally implement?
3. Why can official 29.1 not be presented as your result?
4. How is a raw trajectory transformed into SFT input?
5. Which tokens receive SFT loss, and why are observations masked?
6. What breaks if a tool response is accidentally included in the loss?
7. Why split before resampling?
8. Why compare S0/S1 at equal assistant-token budget?
9. What bias does 1x/2x/5x duplication introduce?
10. How did you calibrate the correctness judge?
11. Derive PPO clipping and the GRPO group baseline.
12. What is advantage collapse in an all-zero rollout group?
13. Derive `IG_t = log P(g|h_t) - log P(g|h_{t-1})`.
14. Why average ground-truth token log-probabilities?
15. Is IG reward potential-based shaping, and can it change the optimum?
16. Why compute IG only around visit turns in the default profile?
17. Why normalize IG and outcome rewards separately?
18. What does IG-Scale do when outcome rewards collapse?
19. At gamma 0.95, how much terminal signal reaches turn 100 or 200?
20. Why can a tool-response position carry advantage but not policy loss?
21. How can decode/re-encode cause BPE boundary drift?
22. What numerical differences can occur between vLLM and FSDP log-probs?
23. How do FSDP, TP, SP, offload, and KV cache interact in this run?
24. How do record/replay tools improve fairness, and what bias remains?
25. What failed, how did you diagnose it, and what would you try with 10x budget?
