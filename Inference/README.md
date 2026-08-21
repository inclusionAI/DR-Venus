# Inference Quick Start

This guide shows how to set up the environment and run inference.



### 1. Environment Setup
- Recommended Python version: **3.12.0** (using other versions may cause dependency issues).

```bash
# Example with Conda
conda create -n react_infer_env python=3.12.0
conda activate react_infer_env
```

### 2. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

#### Environment Configuration

Copy `.env.example` to the untracked `.env` file and configure the API keys
and tool-related settings:

- **SERPER_KEY_ID**: Get your key from [Serper.dev](https://serper.dev/) for web search and Google Scholar
- **JINA_API_KEYS**: Get your key from [Jina.ai](https://jina.ai/) for web page reading
- **API_KEY/API_BASE**: OpenAI-compatible API for page summarization from [OpenAI](https://platform.openai.com/)
- **Proxy(Optional)**: Configure the proxy settings
- **TOOL_CACHE_MODE/TOOL_CACHE_PATH**: use `record` while freezing tool
  responses and `replay` for the final reproducible evaluation
- **TOOL_TIMEOUT_SECONDS**: one end-to-end deadline shared by every HTTP
  retry within a search or visit call


### 4. Run the Inference Script
You can configure `run_demo.sh` with environment variables.
- **MODEL_PATH**: Path to your model weights
- **QUESTION**: Run one backward-compatible ad-hoc question
- **INPUT_FILE**: Run a parquet/JSONL evaluation cohort instead of `QUESTION`
- **OUTPUT_FILE**: Directory for saving results
- **SEED/SAMPLES_PER_QUESTION**: fixed sampling evidence (set the latter to
  four only for the fixed Pass@4 subset)
- **TOOL_CACHE_MODE/TOOL_CACHE_PATH/TOOL_TIMEOUT**: tool record/replay and
  timeout controls
- **TEMPERATURE/TOP_P**: generation sampling controls, also written to each
  raw evidence row
- **MAX_TOKENS_PER_TURN**: per-turn cap (1024 in the fixed protocol); the
  runner further reduces it when the remaining model context is smaller
- **CHECKPOINT_REVISION**: immutable remote model commit/tag. Local checkpoints
  are content-hashed automatically for cohort evaluation
- **REUSE_SERVER**: defaults to `false`; enabling it still requires the
  existing vLLM endpoint to match model root, context length, and revision
- **SERVER_START_TIMEOUT**: readiness deadline (default 900 seconds); an early
  vLLM exit/OOM fails fast and all spawned processes are cleaned up

The low-level wrapper defaults to a safe dry run. The following command only
prints the resolved invocation:

```bash
bash run_demo.sh
```

For example, one vLLM deployment can process the whole fixed cohort and write
resumable raw trajectories:

```bash
MODEL_PATH=/models/dr-venus \
INPUT_FILE=../evaluation/data/eval-core-100.parquet \
OUTPUT_FILE=../artifacts/eval/raw.jsonl \
TOOL_CACHE_MODE=replay \
TOOL_CACHE_PATH=../artifacts/tool_cache/eval.jsonl \
SEED=42 SAMPLES_PER_QUESTION=1 \
DRY_RUN=false CONFIRM_SPEND=YES bash run_demo.sh
```

`CONFIRM_SPEND=YES` is required for every live GPU run. Use the higher-level
`evaluation/run_fixed_eval.sh` for final evidence because it additionally
locks the cohort, requires pilot-budget approval, replays both caches, runs the
Judge, and writes the manifest/report.

Only rows whose termination starts with `answered` are skipped on resume.
Failed rows are retried and remain in the append-only raw audit trail.
The resume fingerprint binds the cohort, sampling settings, checkpoint
identity, and frozen replay-cache SHA-256. Replay cache misses or recorded
provider errors fail the run instead of silently becoming normal Agent text.
An existing raw file with another fingerprint is rejected rather than mixed.
`input_tokens`/`output_tokens` are cumulative inference usage across every LLM
attempt (including discarded retries); final-transcript counts and context
length are recorded separately. Oversized tool results are conditioned on a
head+tail truncation with the removed-token count preserved in evidence.

#### Web Demo (Optional)

If you prefer a web-based interface for visualizing and interacting with DeepResearch, you can launch the web demo with a single command:

```bash
bash run_web_demo.sh
```

### Acknowledgements

The tool-call in this project is partially inspired by the **[Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch/tree/main)** work.
