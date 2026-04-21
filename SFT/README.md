# DR-Venus SFT

SFT training repo built on top of `verl`.

This repo keeps only the pieces needed for supervised fine-tuning and checkpoint merging:

- `verl/`: training implementation
- `train_sft.sh`: main SFT entrypoint
- `run.sh`: root-level convenience wrapper
- `sft_shells/run.sh`: preset wrapper with long-context defaults
- `data_clean/`: convert Xiaohongshu (RED) trajectories into SFT-ready data and run cleaning
- `scripts/merge_checkpoint.sh`: merge sharded checkpoints into Hugging Face format

## Layout

```text
.
├── train_sft.sh
├── run.sh
├── sft_shells/
│   └── run.sh
├── scripts/
│   ├── merge_checkpoint.sh
│   └── model_merger.py
├── data_clean/
│   ├── prepare_trajectories.py
│   └── README.md
├── verl/
├── data/
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Install

```bash
git clone <YOUR_REPO_URL>
cd <REPO_DIR>

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

The training path depends on `torch`, `transformers`, `ray`, `hydra-core`, `pandas`, and `pyarrow`. If your cluster requires a custom CUDA / PyTorch stack, install that first.

## Data Cleaning

`data_clean/` contains the preprocessing pipeline used to convert Xiaohongshu (RED) trajectories into the parquet format expected by SFT training.

Recommended entrypoint:

```bash
python data_clean/prepare_trajectories.py \
  --input <raw_parquet_or_dir> \
  --output-dir <output_dir>
```

What it does:

- normalize raw trajectories into the target SFT schema
- standardize `search` and `visit` tool calls and parse search responses
- remove duplicate tool-call turns
- validate conversation structure
- resample long trajectories by turn count

Main outputs:

- `cleaned_resampled.parquet`: final SFT-ready dataset
- `stats.json`: cleaning statistics and drop reasons

For more CLI options and intermediate artifacts, see `data_clean/README.md`.

## Expected Data

Default mode is multi-turn SFT.

Minimum expected parquet schema:

- `messages`

Common optional fields:

- `tools`
- `enable_thinking`
- `question`
- `gt`

If you want single-turn SFT instead:

```bash
export MULTITURN=False
export PROMPT_KEY=<prompt_column>
export RESPONSE_KEY=<response_column>
```

## Quick Start

### 1. Main entrypoint

```bash
export MODEL_PATH=<MODEL_PATH_OR_HF_ID>
export TRAIN_FILES=<TRAIN_PARQUET_OR_DIR>
export VAL_FILES=<VAL_PARQUET_OR_DIR>
export EXP_NAME=<EXPERIMENT_NAME>

bash train_sft.sh
```

### 2. Preset wrapper

`run.sh` forwards to `sft_shells/run.sh`, which sets a few long-context defaults.

```bash
export DATA_ROOT=<DATA_DIR>
export MODEL_PATH=<MODEL_PATH_OR_HF_ID>

bash run.sh
```

By default the wrapper expands to:

- `TRAIN_FILES=${DATA_ROOT}/train.parquet`
- `VAL_FILES=${DATA_ROOT}/val.parquet`
- `DATA_MAX_LENGTH=200000`
- `SP_SIZE=8`

### 3. Dry run

Use `DRY_RUN=True` to print the final `torchrun` command without launching training.

```bash
export MODEL_PATH=<MODEL_PATH_OR_HF_ID>
export TRAIN_FILES=<TRAIN_PARQUET_OR_DIR>
export VAL_FILES=<VAL_PARQUET_OR_DIR>
export DRY_RUN=True

bash train_sft.sh
```

## Key Environment Variables

Required:

- `MODEL_PATH`
- `TRAIN_FILES`

Most commonly changed:

- `VAL_FILES`
- `PROJECT_NAME`
- `EXP_NAME`
- `NUM_GPUS`
- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `MASTER_PORT`
- `TRAIN_BATCH_SIZE`
- `MICRO_BATCH_SIZE_PER_GPU`
- `DATA_MAX_LENGTH`
- `SP_SIZE`
- `LORA_RANK`
- `LR`
- `TOTAL_EPOCHS`
- `SAVE_FREQ`
- `TEST_FREQ`
- `OUTPUT_ROOT`
- `CKPT_DIR`
- `RESUME_MODE`
- `RESUME_FROM_PATH`

Extra Hydra overrides can be appended directly:

```bash
bash train_sft.sh trainer.save_freq=100 trainer.test_freq=100
```

## Outputs

Default output directory:

```text
./outputs/sft/<EXP_NAME>/
```

Typical contents:

- training logs
- checkpoint directories such as `global_step_*`

## Merge Checkpoints

### FSDP

```bash
bash scripts/merge_checkpoint.sh \
  --backend fsdp \
  --local-dir <CKPT_DIR>/global_step_<STEP>/actor \
  --target-dir <MERGED_MODEL_DIR>
```

### Megatron

```bash
bash scripts/merge_checkpoint.sh \
  --backend megatron \
  --local-dir <CKPT_DIR>/global_step_<STEP>/actor \
  --target-dir <MERGED_MODEL_DIR> \
  --tie-word-embedding
```

Python entrypoint is also available:

```bash
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir <CKPT_DIR>/global_step_<STEP>/actor \
  --target_dir <MERGED_MODEL_DIR>
```

## Notes

- This repo is intentionally scoped to SFT only.
- Paths in examples are placeholders; replace them with your own model, data, and output locations.
- `train_sft.sh --help` prints the full parameter list.

## Acknowledgement

This SFT codebase is built on top of the [veRL](https://github.com/volcengine/verl) framework. We thank the veRL team for open-sourcing the training infrastructure used in this project.
