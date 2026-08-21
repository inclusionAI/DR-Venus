# Budget-matched experiment profiles

These launchers encode the personal-reproduction protocol. They default to a
dry run and cannot start a paid training job until `CONFIRM_SPEND=YES` is set.

## Required order

1. Build deterministic data splits, the equal-token S0/S1 arms, and the fixed
   96-question RL cohort:

   ```bash
   python experiments/prepare_rl_subset.py
   ```

   The builder expects the audited 80,000-row source by default, deduplicates
   normalized questions globally, selects proportionally by `data_source`, and
   writes a hash-bearing manifest beside the output.
2. Run a 3-step pilot (`STEPS=3 ARMS=r2 SEEDS=42`) and record wall time.
3. Extrapolate the cost before any full run:

   ```bash
   python experiments/budget_guard.py \
     --gpu-count 4 \
     --hourly-rate-per-gpu-cny <RATE> \
     --pilot-seconds <SECONDS> \
     --pilot-steps 3 \
     --total-steps 120 \
     --already-spent-cny <SPEND_TO_DATE> \
     --fixed-api-cny <PROJECTED_API_COST> \
     --hard-cap-cny 3000 \
     --output artifacts/budget/rl.json
   ```

   `120` is the total for three RL arms, two seeds, and twenty steps. A nonzero
   exit status means the estimate exceeds the cap.

4. Inspect the dry-run commands from both launchers.
5. Download the starting checkpoint into the GPU environment and compute its
   full directory SHA-256. Formal runs require `MODEL_PATH` to be that local
   directory and require both `MODEL_SHA256=<tree hash>` and
   `MODEL_REVISION=<same tree hash>`; a remote Hugging Face ID is accepted only
   with `ALLOW_CUSTOM_PROTOCOL=YES`.
6. Capture the actual runtime before a formal live run, for example with
   `python3 -m pip freeze --all > artifacts/env/pip-freeze.txt`, and set
   `ENV_LOCK_FILE` to that file. The snapshot stores its hash along with Python,
   package, CUDA, driver, requirements, and optional container-digest evidence.
7. Only then set `DRY_RUN=false CONFIRM_SPEND=YES
   BUDGET_REPORT=artifacts/budget/rl.json API_COST_CNY_PER_RUN=<ACTUAL>` for RL,
   or the corresponding `DRY_RUN=False` values for SFT. Live launchers reject missing reports,
   reports based on anything other than a 3-5 step pilot, mismatched GPU/step
   scopes, over-cap projections, and any hard cap above CNY 3,000.

Dry runs only print commands and do not create evidence files. Every live arm
or seed writes `resolved_command.txt` and `run_snapshot.json` before training,
then finalizes `experiment_manifest.yaml` on success or ordinary failure. The
manifest records the real exit code, elapsed wall time, computed GPU-hours,
log hash, and the newest discoverable `global_step_*` checkpoint. SFT API cost
is fixed at zero; RL requires an explicit finite, non-negative per-run cost.
After each RL run, the shared record-mode tool and Judge caches are copied into
that run directory and hash-bound as immutable final artifacts; use these frozen
copies in `replay` mode when reproducing the run instead of the still-growing
shared cache.

## RL arms

- `r0`: outcome-only GRPO, no turn-level format penalty.
- `r1`: IGPO (`log_prob_diff`, `scaled_separate`, visit-only), no format penalty.
- `r2`: the `r1` configuration plus turn-level format penalty.

All three use the same 32K/exactly-12-turn budget (`TURN_JITTER=0`). The outputs are mechanism-scale
ablations and must not be described as the official 200-turn reproduction.

## Safety rules

- Never place credentials in a command, manifest, cache, or committed `.env`.
- Keep pilots, failed jobs, and cost records; do not delete negative results.
- Stop on NaN/Inf, invalid rollout rate above 10%, or a projected total above
  the approved budget.
