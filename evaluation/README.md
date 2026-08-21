# DR-Venus 可复现评测层

本目录实现预算匹配（budget-matched）的个人评测闭环：固定测试 cohort、校验预测与实验来源、计算指标及置信区间，并同时输出 JSON 和 Markdown。报告会强制展示 `max_turns`、`context_length` 与 `official_comparable=false`，不能被误写成论文官方配置或官方分数。

完整的 Agent → 原始轨迹 → canonical predictions → experiment manifest →
报告链路由 `evaluation/run_fixed_eval.sh` 串联。它默认只打印命令且正式评测默认
`replay`；实际运行必须同时提供 3–5 step pilot 的预算报告和
`CONFIRM_SPEND=YES`。一个 vLLM 实例处理整个 cohort，并按
`(sample_id, seed, sample_index)` 安全续跑。

## 1. 构建 `eval-core-100`

在仓库根目录运行：

```bash
python -m evaluation.build_eval_core \
  --input RL/data/test.parquet \
  --output evaluation/data/eval-core-100.parquet \
  --size 100 \
  --seed 42
```

构建过程按 `data_source` 比例使用最大余数法分配名额，再按 `SHA256(seed, sample_id)` 排序抽样，不依赖 parquet 的行顺序。当前数据应得到：`2wiki/popqa/tq/hotpotqa/nq/musique` 各 16 条，`Bamboogle` 4 条。若来源数不再是 7，命令会失败，防止上游数据漂移被静默接受。

输出旁会生成 `eval-core-100.parquet.manifest.json`，记录输入/输出 SHA-256、抽样算法、seed、来源分布和全部样本 ID。构建及正式 parquet cohort 的证据校验依赖 `pandas` 与 parquet engine（推荐 `pyarrow`）；指标计算本身只使用 Python 标准库。

### `eval-web-60` 与固定 Pass@4 子集

仓库没有随附 BrowseComp/BrowseComp-ZH 原始数据，也不会联网下载或生成替代题。取得合规的两份本地原始文件后，显式传入：

```bash
python -m evaluation.build_eval_web \
  --browsecomp /data/BrowseComp.jsonl \
  --browsecomp-zh /data/BrowseComp-ZH.jsonl \
  --output evaluation/data/eval-web-60.parquet \
  --seed 42
```

输入可为 JSON、JSONL 或 parquet；每行必须有非空问题和答案。优先使用原始稳定 ID；官方文件若没有 ID，则明确用 NFKC 归一化问题的 SHA-256 作为内容 ID。构建器分别按 `SHA256(seed, source, stable_id)` 固定 30 题，拒绝缺字段、重复 ID/问题和任一来源不足 30 条，并写入两个输入及输出的 SHA-256。缺少真实输入时命令直接失败，因此仓库不会出现伪造的 `eval-web-60`。

Pass@4 只在 `eval-core-100` 的固定 20 题子集运行。已提交的 [pass4-20.parquet](data/pass4-20.parquet) 及其 [manifest](data/pass4-20.parquet.manifest.json) 可通过下列命令重建：

```bash
python -m evaluation.build_pass4_subset
```

构建前会校验父 cohort 的 SHA、行数、样本 ID 和父 manifest；再按来源分层并用带 domain separation 的稳定哈希抽 20 题。该子集只用于每题四次生成，不进入 `eval-core-100` 的 overall 指标。

## 2. 预测与实验接口

每一行 `predictions.jsonl` 代表一次生成。规范示例见 [predictions.example.jsonl](examples/predictions.example.jsonl)，必填字段如下：

| 字段 | 含义 |
|---|---|
| `sample_id`, `data_source` | cohort 中的稳定问题 ID 与来源 |
| `model`, `seed`, `sample_index` | 模型和一次生成的唯一键；同题多次生成用于 Pass@K |
| `answer`, `references`, `score` | 抽取后的答案、可接受答案数组、`[0,1]` Judge 分；未 Judge 时 `score=null` |
| `completed`, `format_valid`, `turns` | 是否正常结束、格式是否合法、实际轮数 |
| `tool_calls`, `browse_calls`, `timeout_count` | 工具调用及超时证据 |
| `input_tokens`, `output_tokens`, `latency_seconds`, `cost_cny` | 资源与成本证据 |
| `max_turns`, `context_length` | 本次实际评测上限，必须与 manifest 完全一致 |
| `trajectory_path`, `metadata` | 原始轨迹位置（允许 `null`）及扩展元数据 |

`references` 应在生成预测文件时把原 parquet 中以 `<|answer_split|>` 分隔的答案拆成数组。未知顶层字段、重复 `(sample_id, seed, sample_index)`、越界计数和不一致的限制均会被拒绝；自定义 Judge 信息放进 `metadata`。

### 可选 Judge：默认离线回放

Judge CLI 读写相同的 canonical prediction schema，把 `[0,1]` 分数写入 `score`，并在 `metadata.judge` 记录 provider、model、prompt version、cache key、命中状态、理由、响应 hash、时间和 token usage。默认 `offline` 只读缓存，cache miss 会终止，绝不会隐式请求网络：

```bash
python -m evaluation.judge_predictions predictions.jsonl scored.jsonl \
  --mode offline \
  --model "$JUDGE_MODEL" \
  --cache artifacts/evaluation/judge_cache.jsonl
```

首次采集必须显式切换 `online`。CLI 不接受明文 API key 参数，只从指定环境变量读取；请求失败或非 JSON 输出会按上限重试并最终 fail-closed：

```bash
export JUDGE_BASE_URL=https://provider.example/v1
export JUDGE_MODEL=judge-model-id
export JUDGE_API_KEY=replace-me
python -m evaluation.judge_predictions predictions.jsonl scored.jsonl \
  --mode online --retries 3 --timeout-seconds 60
```

正式实验应锁定 Judge model 和 prompt version，并保存 cache。不要把真实 key、带敏感内容的临时文件或未脱敏响应提交到仓库。

### 100 条人工校准（盲审）

先从已 Judge 的 `sample_index=0` 预测中稳定抽取 100 条。模板故意不含 Judge 分数，降低人工标签被锚定的风险：

```bash
python -m evaluation.judge_calibration prepare \
  --predictions scored.jsonl \
  --output artifacts/evaluation/judge-review-100.jsonl \
  --seed 42
```

人工只填写每行的 `human_score`（正确为 `1`，错误为 `0`）和可选 `notes`，不要修改其他字段。配套 manifest 保存原 prediction hash、全部 review ID 和不可编辑字段 hash。完成后计算一致率、Wilson 95% CI、Cohen's kappa、混淆矩阵及分歧清单：

```bash
python -m evaluation.judge_calibration score \
  --template artifacts/evaluation/judge-review-100.jsonl \
  --manifest artifacts/evaluation/judge-review-100.jsonl.manifest.json \
  --predictions scored.jsonl \
  --output artifacts/evaluation/judge-calibration.json
```

默认以 Judge score `>=0.5` 判正，目标一致率为 90%。工具会校验恰好 100 条均已填写，且原 predictions 或模板不可编辑字段未被换包；未达标时应报告噪声与分歧类型，而不是调整人工标签。

`experiment_manifest.yaml` 的规范示例见 [experiment_manifest.example.yaml](examples/experiment_manifest.example.yaml)。文件采用 JSON 语法（JSON 是合法的 YAML 1.2），所以无需 YAML 依赖也能验证。必须记录：

- 上游 commit、checkpoint、模型与带 SHA-256 的固定评测集；
- seeds、GPU、Python/平台、`max_turns/context_length`；
- GPU-hour、API 人民币成本、配置与产物路径；
- `protocol.official_comparable=false`。校验器不会接受 `true`。

`evaluation_dataset.path` 和 `artifacts.*.path` 都相对 `experiment_manifest.yaml` 所在目录解析。提供 manifest 时，评测器默认按正式证据 fail-closed：除 cohort 的文件、SHA-256、行数与语义外，还验证当前 predictions 正是 manifest 指向的文件，逐一验证已声明的 raw/tool/Judge cache 文件与哈希，并把每条 canonical prediction 的 run fingerprint、checkpoint revision/SHA、tool-cache SHA、采样 seed 与生成参数交叉绑定到 manifest。Judge score 还会与 Judge model、prompt/provider 及缓存记录核对；任一项不一致都不会生成报告。该行为不信任 `protocol.label`。

## 3. 生成评测报告

```bash
python -m evaluation.evaluate predictions.jsonl \
  --manifest experiment_manifest.yaml \
  --output-json results/evaluation.json \
  --output-markdown results/evaluation.md \
  --bootstrap-seed 42 \
  --bootstrap-samples 10000 \
  --pass-k 1 4
```

如需对候选模型和基线做相同问题、相同生成序号的 paired bootstrap：

```bash
python -m evaluation.evaluate candidate.jsonl \
  --manifest candidate_manifest.yaml \
  --baseline baseline.jsonl
```

主指标、按来源指标、普通 bootstrap CI 和 paired CI 只使用 `sample_index=0`；Pass@K 单独使用全部 generation，避免 Pass@4 子集在 overall 中被四倍加权。paired 比较要求基线与候选的 index-0 `(sample_id, seed)`、reference、来源及推理限制一致；其他 generation 不进入 paired CI。bootstrap 以 `sample_id` 为重采样单位并固定 seed。

报告包含：

- answer accuracy（Judge 全覆盖时用 Judge 分，否则明确标注 EM fallback）、EM、token F1；
- 完成率、格式率、turn 均值/p95、browse/tool-call ratio、含超时样本比例；
- token、延迟、人民币成本的总量/均值与必要的 p95；
- answer accuracy、EM、F1 的 percentile bootstrap 95% CI；
- 标准无偏估计 `pass@k = 1 - C(n-c,k)/C(n,k)`，并报告具备至少 K 次生成的问题覆盖率；
- 按 `data_source` 分组指标以及可选的 candidate-minus-baseline paired CI。

EM/F1 采用大小写、冠词、标点和空白归一化；中文按单字 token 计算 F1。`browse_ratio` 定义为 `sum(browse_calls) / sum(tool_calls)`，`timeout_rate` 定义为至少发生一次超时的生成占比。

## 4. 验证

```bash
python -m pytest -q \
  tests/test_evaluation_build.py \
  tests/test_evaluation_metrics.py \
  tests/test_evaluation_schemas.py \
  tests/test_eval_protocols.py
```

以下命令可直接验证示例 cohort、manifest 和 prediction 的完整闭环：

```bash
python -m evaluation.evaluate \
  evaluation/examples/predictions.example.jsonl \
  --manifest evaluation/examples/experiment_manifest.example.yaml \
  --allow-legacy-manifest
```

`--allow-legacy-manifest` 只为旧示例提供显式的 development-only 降级，输出不会被标为已验证的正式证据。示例 cohort 的 hash 是真实值；manifest 中的上游 commit、checkpoint 和使用量仅为示范，正式实验必须替换为个人实验的真实证据。
