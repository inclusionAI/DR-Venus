# REDSearcher_SFT_10K

One-click cleaning pipeline for REDSearcher-style trajectories.

The repository now exposes a single recommended entrypoint:

```bash
python prepare_trajectories.py --input <raw_parquet_or_dir> --output-dir <output_dir>
```

## What the pipeline does

Given raw trajectory data, the pipeline will:

1. Normalize raw trajectories into the target SFT schema.
2. Standardize `search` and `visit` tool calls.
3. Parse Google-style `search` tool responses into structured JSON.
4. Remove duplicate `search` and `visit` tool turns.
5. Validate the final conversation structure.
6. Resample long trajectories based on turn count.

Default resampling policy:

- `turns <= 50`: `1x`
- `50 < turns <= 100`: `2x`
- `turns > 100`: `5x`

These thresholds and multipliers are configurable from the CLI.

## Repository Layout

```text
REDSearcher_SFT_10K/
├── prepare_trajectories.py          # Recommended one-click pipeline
├── data/                            # Raw parquet shards and legacy outputs
├── data_clean_correct/      # Existing reference subset
└── README.md
```

## Input Schemas

The pipeline supports two input layouts.

### Raw input

Expected columns:

- `meta.question`
- `meta.answer`
- `messages`
- `system_prompt` (optional if you use `--system-prompt-mode row`)

### Already-clean input

Expected columns:

- `question`
- `messages`
- `gt`

## Quick Start

### 1. Install dependencies

```bash
python -m pip install pandas pyarrow numpy json5
```

### 2. Run on the raw dataset directory

```bash
python prepare_trajectories.py \
  --input <raw_parquet_or_dir> \
  --output-dir <output_dir> \
  --system-prompt-mode template \
  --current-date 2026-03-01 \
  --shuffle \
  --seed 42
```

## Output Artifacts

Each run writes:

- `cleaned_resampled.parquet`: final cleaned dataset
- `stats.json`: row counts, drop reasons, duplicate-removal stats, turn buckets, resampling distribution

Optional outputs:

- `stage1_normalized.parquet`: output after raw normalization
- `stage2_deduped.parquet`: output after duplicate tool-call removal
- `cleaned_resampled.jsonl`: JSONL export of the final dataset

These optional files are enabled with:

```bash
--keep-intermediate
--write-jsonl
```

## Useful CLI Options

```bash
python prepare_trajectories.py --help
```

Most useful flags:

- `--input`: raw parquet file or directory
- `--output-dir`: destination directory
- `--file-pattern`: glob used when input is a directory, default `train-*.parquet`
- `--system-prompt-mode`: `template` or `row`
- `--current-date`: append a deterministic date to the bundled system prompt
- `--short-max-turns`: default `50`
- `--medium-max-turns`: default `100`
- `--short-multiplier`: default `1`
- `--medium-multiplier`: default `2`
- `--long-multiplier`: default `5`
- `--shuffle`: shuffle final rows after resampling
- `--seed`: shuffle seed
- `--keep-intermediate`: save stage-level parquet outputs
- `--keep-metadata`: keep debug columns like source row, turns, and sample copy id
- `--write-jsonl`: also export the final dataset as JSONL

## Validation Notes

The pipeline validates that:

- conversations start with a `system` message
- roles alternate correctly after the system turn
- non-initial user turns contain `<tool_response>...</tool_response>`
- intermediate assistant turns contain `<think>` and `<tool_call>`
- final assistant turns contain `<think>` and `<answer>`

Rows that fail validation are dropped and recorded in `stats.json`.