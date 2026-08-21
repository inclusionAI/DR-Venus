# DR-Venus SFT data governance

`prepare_trajectories.py` is the recommended entrypoint for turning raw REDSearcher-style trajectories into leakage-free SFT splits.

```bash
python prepare_trajectories.py --input <raw_parquet_or_dir> --output-dir <output_dir>
```

The default is fully offline: `--correctness-mode none` never calls a service. Correctness filtering is enabled only when a field, an existing cache, or an explicitly configured judge is selected.

## Pipeline order

The order is fixed so validation/test examples can never be multiplied into training:

1. Normalize either supported input schema.
2. Remove disallowed and duplicate tool-call/response pairs.
3. Validate roles and `<think>`, `<tool_call>`, `<tool_response>`, and `<answer>` tags.
4. Optionally filter final-answer correctness.
5. Group by normalized question and deterministically split 90/5/5.
6. Apply turn-aware 1×/2×/5× resampling to the training split only.
7. Write logical/file hashes and provenance to `data_manifest.json`.

Question grouping uses `NFKC + whitespace collapse + casefold`. Groups are ordered by a seed-salted SHA-256 digest and allocated with deterministic ratio quotas. With at least three question groups, every non-zero split receives a group; this intentionally favors leakage prevention and usable validation/test data over exact ratios for tiny smoke datasets.

Default resampling policy:

- `turns <= 50`: 1×
- `50 < turns <= 100`: 2×
- `turns > 100`: 5×

## Input schemas

Raw rows contain:

- `meta.question`
- `meta.answer`
- `messages`
- `system_prompt` (optional)

Already-clean rows contain:

- `question`
- `messages`
- `gt`

## Correctness modes

### Offline field

Values may be booleans, numeric scores, common strings (`correct`, `incorrect`, `pass`, `fail`), or objects containing `correct`, `is_correct`, `score`, `reward`, or `judgement`. Dotted paths are supported.

```bash
python prepare_trajectories.py \
  --input <input> \
  --output-dir <output> \
  --correctness-mode field \
  --correctness-field meta.correctness \
  --correctness-threshold 0.5
```

Missing and invalid field values are dropped under separate reasons in `stats.json`.

### Strictly offline cache

Cache keys bind the exact question, reference answer, candidate `<answer>`, and prompt version. A JSONL entry has this form:

```json
{"cache_key":"<sha256>","correct":true,"score":1.0,"prompt_version":"dr-venus-correctness-v1"}
```

JSON arrays and JSON objects keyed by `cache_key` are also accepted. Cache misses and malformed decisions are dropped, never sent over the network.
Entries without an explicit key may instead provide `question`, `gt` (or `reference`), and `candidate` (or `prediction`); the loader derives the same versioned key. Judge-mode write-through records retain these three inputs plus the model and short reason so decisions can be manually calibrated later.
Use a separate cache when intentionally changing judge model or prompt policy. The manifest inventories the model/prompt provenance of cache records actually used in the run.

```bash
python prepare_trajectories.py \
  --input <input> \
  --output-dir <output> \
  --correctness-mode cache \
  --judge-cache <decisions.jsonl>
```

### Explicit OpenAI-compatible judge

Only `judge` mode permits HTTP(S). It reads the cache first and writes new decisions through to it. Literal secrets are not accepted as CLI options; the key is read from the named environment variable and is never written to logs or manifests.

```bash
export OPENAI_API_KEY='<secret>'
python prepare_trajectories.py \
  --input <input> \
  --output-dir <output> \
  --correctness-mode judge \
  --judge-cache <decisions.jsonl> \
  --judge-base-url https://provider.example/v1 \
  --judge-model <model> \
  --judge-retries 2
```

The judge must return JSON containing a recognized correctness value, preferably `{"correct": true|false, "reason": "..."}`. Exhausted network or parser retries fail the run instead of silently labeling a trajectory incorrect.

## Outputs

Every successful run writes:

- `train_clean.parquet`: filtered, unresampled training rows; use this to construct controlled SFT arms.
- `train_resampled.parquet`: only the training split after 1×/2×/5× expansion.
- `validation.parquet`: unresampled validation rows.
- `test.parquet`: unresampled held-out rows.
- `cleaned_resampled.parquet`: compatibility alias containing the same logical rows as `train_resampled.parquet`.
- `stats.json`: drop reasons, tool/dedup counts, split counts, and resampling statistics.
- `data_manifest.json`: source hashes/counts, split seed/algorithm, judge/prompt provenance, output file hashes, and logical dataset hashes.

`--write-jsonl` adds matching JSONL artifacts. `--keep-intermediate` writes normalization and post-filter stage parquet files. `--keep-metadata` retains source row, turn count, correctness provenance, and resample copy columns; default training outputs remain `question/messages/gt` only.

Use explicit, separate paths when training:

```bash
TRAIN_FILES=<output>/train_resampled.parquet
VAL_FILES=<output>/validation.parquet
```

Do not point validation at `cleaned_resampled.parquet` or the training file.

## Token-budget-matched S0/S1 arms

`build_sft_arms.py` builds the planned uniform S0 and turn-weighted S1 datasets from `train_clean.parquet`:

```bash
python build_sft_arms.py \
  --input <output>/train_clean.parquet \
  --data-manifest <output>/data_manifest.json \
  --output-dir <arms_output> \
  --assistant-token-budget 20000000 \
  --seed 42 \
  --tokenizer Qwen/Qwen3-4B-Thinking-2507 \
  --max-length 32768 --truncation right \
  --train-batch-size 8
```

Outputs are `s0_uniform.parquet`, `s1_turn_weighted.parquet`, and `sft_arms_manifest.json`. Both arms draw complete trajectories with replacement toward the same assistant-token target. With a tokenizer, counts mirror the training `MultiTurnSFTDataset` loss masks and its 32K right truncation: generation-prompt tokens are excluded while retained assistant suffix and template-terminator tokens are included. Sampling proceeds in complete global train batches (default 8), so the trainer's `drop_last=True` consumes every selected row. The builder jointly searches alternative final batches and fails unless the two realized supervised-token totals differ by at most 0.1% of the target. Target, realized/effective totals, between-arm gap/tolerance, overshoot, selection hashes, source exposure counts, tokenizer, max length, truncation, and seed are recorded.

Without `--tokenizer`, the builder uses a deterministic approximation intended only for tests and cheap pilots. Formal experiment manifests should use the exact training tokenizer.

## Tests

From the repository root:

```bash
python -m unittest tests.test_data_pipeline -v
```

Coverage includes both input schemas, illegal/duplicate tool pair removal, role/tag validation, more than 25 correctness encodings, field/cache decisions, judge parsing and retries with a fake transport, deterministic question isolation, train-only resampling, manifest hashes, and injected token counting for S0/S1 construction. Tests do not access the network.
