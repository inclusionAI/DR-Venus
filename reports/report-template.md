# DR-Venus budget-matched reproduction report

## 1. Claim and attribution

State upstream projects, personal modifications, exact commit, and what is not
claimed.

## 2. Reproduction question

Define the SFT sampling and RL reward hypotheses before showing results.

## 3. Data

Include source hashes, correctness filtering, 90/5/5 split, leakage checks,
turn/tool distributions, and the 100-item human judge audit.

## 4. Methods and budgets

Report model/checkpoint hashes, context, max turns, samples per prompt, seeds,
hardware, GPU-hours, API calls, and spend. Put personal and official budgets in
separate columns.

## 5. SFT results

Compare base, S0, and S1. Include held-out NLL, task metrics, format rate,
uncertainty, and failures.

## 6. RL results

Compare official-SFT start, R0, R1, and R2 across both seeds. Include outcome
diversity, IG activation/distribution, KL, turns, browse ratio, and cost.

## 7. Evaluation and sensitivity

Explain live versus replay tools, judge calibration, Pass@1/Pass@4, paired
bootstrap, and any CI crossing zero.

## 8. Failure analysis and limitations

Include negative runs, web drift, reward hacking, truncation, and the gap from
the official 200K/256K/200-turn setting.

## 9. Conclusions

Only claim effects supported by manifests and raw predictions.
