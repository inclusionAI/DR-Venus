# DR-Venus 训练全流程梳理：数据 → SFT → RL

> 对照论文 *DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data*（arXiv:2604.19859）与本仓库代码整理。
> 文中 `file:line` 均为仓库内相对路径。论文里的工具叫 `browse`，代码里叫 `visit`，二者是同一个工具。

---

## 0. 总览

| 项 | 取值 |
|---|---|
| 基座模型 | `Qwen3-4B-Thinking-2507` |
| 工具 | `search`（Serper，Google 搜索）+ `visit`（Jina Reader 抓页面，再由 LLM 按 goal 摘要） |
| SFT 数据 | REDSearcher SFT 10K 轨迹 → 清洗/过滤后 9,365 条 → 按轮数重采样后 18,745 条 |
| RL 数据 | REDSearcher RL 1K 问答对（`Zchu/REDSearcher_RL_1K`） |
| RL 算法 | IGPO（基于信息增益的 GRPO 变体）+ 轮级格式惩罚 |
| 交互步数 | 最多 200 轮工具调用 |
| 上下文 | 训练 200K+（SFT 200K，RL 约 256K） |

```
REDSearcher 原始轨迹 (10,001)
   │  SFT/data_clean/prepare_trajectories.py
   │    ① 格式对齐  ② 非法工具裁剪  ③ 重复调用去重  ④ 结构校验
   │  (+ 外部 LLM-judge 正确性过滤 → 9,365)
   │    ⑤ 按轮数重采样 (1x/2x/5x → 18,745)
   ▼
cleaned_resampled.parquet ──► SFT/train_sft.sh (verl fsdp_sft_trainer, 仅 assistant token 计损失)
                                   │
                                   ▼  DR-Venus-4B-SFT
REDSearcher RL 1K (question, gt) ──► RL/train_igpo.sh (verl main_ppo + DRAgentLoop 异步 rollout)
                                   │    奖励 = LLM-judge 结果奖励 + visit 轮信息增益 + 格式惩罚
                                   │    优势 = 分组分别归一化 + IG 缩放 + 轮级折扣(γ=0.95)
                                   ▼  DR-Venus-4B-RL
```

---

## 1. SFT 阶段

### 1.1 数据准备

入口脚本：[SFT/data_clean/prepare_trajectories.py](../SFT/data_clean/prepare_trajectories.py)

```bash
python SFT/data_clean/prepare_trajectories.py \
  --input <raw_parquet_or_dir> --output-dir <out> \
  --system-prompt-mode template --current-date 2026-03-01 --shuffle --seed 42
```

**输入格式**（两种都支持，见 `normalize_row`，[prepare_trajectories.py:484](../SFT/data_clean/prepare_trajectories.py#L484)）：
- 原始 REDSearcher：`meta.question`、`meta.answer`、`messages`、可选 `system_prompt`
- 已清洗：`question`、`messages`、`gt`

**流水线步骤**（对应论文 "Structural Cleaning → Correctness Filtering → Turn-Aware Resampling"）：

| 步骤 | 做了什么 | 代码位置 | 论文统计 |
|---|---|---|---|
| ① 环境对齐 / 格式规范化 | 统一替换成推理时使用的 system prompt（`DEFAULT_SYSTEM_PROMPT`，含 search/visit 两个工具的 schema）；首条 user 为问题；中间 assistant 轮规范成 `<think>…</think>\n<tool_call>{json}</tool_call>`；最后一轮规范成 `<think>…</think>\n<answer>…</answer>`；tool 返回统一包成 `<tool_response>`，其中 Google 搜索的 markdown 结果被解析为结构化 JSON（`query` + `web_page_info_list[title,url,snippet]`）；修复 UTF-16 代理字符 | `normalize_raw_messages` [:402](../SFT/data_clean/prepare_trajectories.py#L402)、`clean_toolcall_output` [:332](../SFT/data_clean/prepare_trajectories.py#L332)、`clean_answer_output` [:369](../SFT/data_clean/prepare_trajectories.py#L369)、`parse_search_results` [:273](../SFT/data_clean/prepare_trajectories.py#L273) | 10,001 条全部转换成功 |
| ② 工具参数统一 | `search` 的 `query/queries` 统一为 `{"query": [..]}`；`visit` 统一为 `{"url": [..], "goal": str}` | `normalize_tool_call` [:290](../SFT/data_clean/prepare_trajectories.py#L290) | — |
| ③ 非法工具裁剪 | 只保留 `search`/`visit`；其它工具（主要是 PythonInterpreter）的调用轮连同其后的 tool_response 一起删除 | `prune_disallowed_tool_turns` [:541](../SFT/data_clean/prepare_trajectories.py#L541) | 涉及 1,064 条轨迹 / 3,378 次调用 |
| ④ 重复调用去重 | 以 (工具名, 排序去重后的 query/url 集合) 作为签名；若某个 assistant 轮的所有调用签名都已出现过，则删除该轮及其 tool_response | `canonical_signature` [:590](../SFT/data_clean/prepare_trajectories.py#L590)、`remove_duplicate_tool_calls` [:618](../SFT/data_clean/prepare_trajectories.py#L618) | 删除 15,728 次重复交互（主要是重复 browse），10,000 条有效 |
| ⑤ 结构校验 | system 开头；user/assistant 严格交替；非首个 user 必须含 `<tool_response>`；中间 assistant 必须有 `<think>`+`<tool_call>` 且不能有 `<answer>`；最后一个 assistant 必须有 `<think>`+`<answer>` 且不能有 `<tool_call>`。不合格的丢弃并记录原因到 `stats.json` | `validate_messages` [:685](../SFT/data_clean/prepare_trajectories.py#L685) | — |
| ⑥ 正确性过滤 | 用 Qwen3-235B-A22B-Instruct-2507 作为 judge，判断最终答案是否正确，只保留正确的轨迹 | **不在此脚本中**（论文中的离线步骤，实现参考 Tongyi-DeepResearch） | 保留 9,365 条（有效轨迹的 93.65%） |
| ⑦ 按轮数重采样 | 轮数 = `(len(messages)-1)//2`；≤50 轮 1×，51–100 轮 2×，>100 轮 5×；可 `--shuffle` | `compute_multiplier` [:723](../SFT/data_clean/prepare_trajectories.py#L723)、`main` 中的重采样循环 [:875](../SFT/data_clean/prepare_trajectories.py#L875) | 9,365 → 18,745；>50 轮占比 60.28%→80.15%，>100 轮占比 13.29%→33.21% |

**输出**：`cleaned_resampled.parquet`（列 `question / messages / gt`）+ `stats.json`；可选 `--keep-intermediate`（stage1/stage2 parquet）、`--write-jsonl`。

> 注意：如果要严格复现论文，正确性过滤需要在 ⑤ 之后、⑦ 之前自行用 LLM judge 对 `gt` 与最终 `<answer>` 做一次判定。

### 1.2 SFT 训练

入口：[SFT/run.sh](../SFT/run.sh) → [SFT/sft_shells/run.sh](../SFT/sft_shells/run.sh)（预设长上下文参数）→ [SFT/train_sft.sh](../SFT/train_sft.sh) → `torchrun -m verl.trainer.fsdp_sft_trainer`

```bash
export DATA_ROOT=<dir containing train.parquet / val.parquet>
export MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507
bash SFT/run.sh
```

**数据加载**：多轮模式 `MultiTurnSFTDataset`（[SFT/verl/utils/dataset/multiturn_sft_dataset.py](../SFT/verl/utils/dataset/multiturn_sft_dataset.py)）
- 对 `messages` 套用 Qwen3 chat template，逐条消息分段 tokenize；
- **只有 assistant 消息的内容 token 计入 loss**（`loss_mask=1`），system / user / tool_response 以及 assistant 的生成前缀 `<|im_start|>assistant\n` 都是 0（[multiturn_sft_dataset.py:174-193](../SFT/verl/utils/dataset/multiturn_sft_dataset.py#L174-L193)）；
- 超过 `max_length` 时右截断（`TRUNCATION=right`）。

对应论文的 SFT 目标：

$$\mathcal{L}_{\mathrm{SFT}}(\theta)=-\sum_{H\in\mathcal{D}_{\mathrm{SFT}}}\sum_{i\in\mathcal{M}(H)}\log\pi_\theta(x_i\mid x_{<i})$$

其中 $\mathcal{M}(H)$ 是轨迹 $H$ 中 assistant token 的位置集合，观测（工具返回）不监督。

**训练器**：[SFT/verl/trainer/fsdp_sft_trainer.py](../SFT/verl/trainer/fsdp_sft_trainer.py)，loss 为 masked token-level CE（[fsdp_sft_trainer.py:369](../SFT/verl/trainer/fsdp_sft_trainer.py#L369)）。

**超参**（`sft_shells/run.sh` + `train_sft.sh` + `verl/trainer/config/sft_trainer.yaml` 默认值，与论文一致）：

| 参数 | 值 |
|---|---|
| lr | 1e-5，cosine 调度，warmup 比例 0.1 |
| 优化器 | AdamW，betas (0.9, 0.95)，weight decay 0.01，grad clip 1.0 |
| 全局 batch / 每卡 micro batch | 32 / 1 |
| epoch | 1 |
| 最大长度 | 200,000 token（`train_sft.sh` 自身默认是 32,768，需用 `sft_shells/run.sh` 或手动设置） |
| 序列并行 | Ulysses SP=8 |
| 其它 | FSDP2、gradient checkpointing、remove padding、全参（`LORA_RANK=0`） |
| 硬件 | 8×A100 |

**产物合并**：[SFT/scripts/merge_checkpoint.sh](../SFT/scripts/merge_checkpoint.sh) 把 `global_step_*/actor` 分片合并成 HF 格式，作为 RL 的起点（`DR-Venus-4B-SFT`）。

**消融**：重采样带来 BrowseComp +4.0（22.8→26.8）、BrowseComp-ZH +1.8（33.9→35.7）。

---

## 2. RL 阶段

### 2.1 数据准备

RL 只需要「问题 + 标准答案」，不需要轨迹。parquet 格式（verl RLHF 格式）：

| 列 | 内容 |
|---|---|
| `prompt` | `[{"role": "user", "content": question}]` |
| `reward_model` | `{"ground_truth": str 或 List[str], "style": ...}` |
| `data_source` | 数据来源标签，如 `nq` / `2wiki` / `browse_comp` |
| `ability` / `extra_info` | 辅助信息 |

- 论文：使用 `Zchu/REDSearcher_RL_1K` 的 1K 问答对，论文未提及难度/通过率过滤。
- **仓库自带的 [RL/data/train.parquet](../RL/data/train.parquet) 不是论文数据**：它有 80,000 条，来自 NQ / 2Wiki / TriviaQA（`test.parquet` 3,197 条，NQ/2Wiki/TQ/HotpotQA/PopQA 等），是 IGPO/DeepResearcher 的示例数据。复现论文时需要把 REDSearcher RL 1K 转成上表格式替换。
- 若 `ground_truth` 是列表，最后一个元素为最终答案，前面的元素作为“关键路径”（命中时有小额奖励，见 2.4）。

System prompt 在 rollout 时由 [RL/tool_server/tool_prompt.py](../RL/tool_server/tool_prompt.py) 的 `SYSTEM_PROMPT` 注入（[dr_agent_loop.py:349](../RL/scrl/llm_agent/dr_agent_loop.py#L349)），与 SFT/推理保持同一工具协议。

### 2.2 启动与整体流程

入口：[RL/train_igpo.sh](../RL/train_igpo.sh) → `python -m verl.trainer.main_ppo`，`.env` 里配置 Serper / Jina / 摘要模型 / judge 模型。

每个训练 step（[RL/verl/trainer/ppo/ray_trainer.py](../RL/verl/trainer/ppo/ray_trainer.py)）：

1. **Rollout**：取 16 个问题，每题采样 `agent_grpo.n=8` 条轨迹（GRPO 分组），由 `DRAgentLoop` 异步多轮交互生成；
2. **结果奖励**：LLM judge 判定最终答案；
3. **信息增益**：rollout 结束后，用 FSDP actor 在每个 visit 轮边界计算标准答案的 log-prob（`ig_kv_cache` 阶段）；
4. **格式惩罚**：逐轮检查格式，不合格的轮奖励替换为 −1；
5. **奖励铺到 token 上**：每轮奖励放在该轮 `<|im_end|>` 位置，结果奖励放在最后一个 token；
6. **优势计算**：`grpo_info_gain` 估计器（分别归一化 → IG 缩放 → 轮级折扣）；
7. **策略更新**：PPO clip 目标 + KL loss，tool_response token 不参与 loss。

### 2.3 Rollout：`DRAgentLoop`

代码：[RL/scrl/llm_agent/dr_agent_loop.py](../RL/scrl/llm_agent/dr_agent_loop.py)（配置 [RL/configs/dr_agent_loop.yaml](../RL/configs/dr_agent_loop.yaml)），vLLM 推理（TP=4），temperature 1.0、top_p 0.95。

- 每一轮：对当前 messages 套模板 → 生成（单轮最多 8,192 token）→ 解析 `<tool_call>` → 调用 search / visit（超时 150s）→ 把结果作为 `<tool_response>` 的 user 消息追加 → 下一轮；
- 最大轮数：`max_turns=200`，并对每条样本加 [−5, +5) 的随机抖动（[dr_agent_loop.py:357](../RL/scrl/llm_agent/dr_agent_loop.py#L357)）；
- **强制收尾**（[dr_agent_loop.py:446-530](../RL/scrl/llm_agent/dr_agent_loop.py#L446-L530)）：上下文超过 `max_model_len(261000) - 8192`、轮数到上限、或异步 `completion_cutoff=0.9`（90% 样本已完成时截断拖尾样本）时，把最后一条 user 消息替换为 “Provide your final answer in `<answer></answer>` tags now.”；
- 每轮记录 `turn_boundaries`（`step`、`token_len`、`msg_count`、`tool_name`），供后续 IG 与格式检查定位；
- `visit` 工具：Jina 抓取正文，截断到约 95K token，交给摘要 LLM 按 goal 提取 evidence/summary（[RL/tool_server/tool_visit.py](../RL/tool_server/tool_visit.py)）。

### 2.4 奖励函数

总体上，一条轨迹 $i$ 共 $T_i$ 轮，前 $T_i-1$ 轮是工具调用轮，第 $T_i$ 轮是回答轮：

| 位置 | 奖励 |
|---|---|
| 回答轮（最后一个 token） | 结果奖励 $r^O$；若回答轮格式错误，替换为 $-\lambda_{fmt}=-1$ |
| visit 轮（轮末 `<|im_end|>`） | 信息增益 $r^{IG}_{i,t}$；若该轮格式错误，替换为 −1 |
| search 等非 IG 轮 | 默认无奖励；若格式错误，则在该轮末尾放 −1 |

#### (1) 结果奖励：LLM-as-Judge

代码：[RL/verl/utils/reward_score/llm_judge.py](../RL/verl/utils/reward_score/llm_judge.py)，`train_reward_type=llm`，reward manager 为 `naive_batch`。

- 从回答中抽取 `<answer>…</answer>`；
- 与 GT **完全相等** → 1.0；为空或为占位文本 → 0.0；
- 否则用 BrowseComp 标准 judge prompt（`JUDGE_PROMPT_BC`）请求 judge 模型（默认 `Qwen3-235B-A22B-Instruct-2507`，temperature 0.1），解析 `correct: yes/no` → 1.0 / 0.0，解析失败记 0；
- 若 GT 是列表且答对，轨迹中每命中一个关键路径元素 +0.1；
- 备选：`TRAIN_REWARD_TYPE=f1` 使用 token-level F1（[format_and_f1.py](../RL/verl/utils/reward_score/format_and_f1.py)），标签不配对或无 `<answer>` 时为 −2；
- 验证集使用 `llm_em_noformatf1`（同时记录 EM 与不含格式惩罚的 F1）。

#### (2) 信息增益奖励（IGPO 的核心）

思路：把“标准答案”当作伪回复，看模型在每一轮新信息到来后，生成标准答案的概率提高了多少。

论文公式（$g$ 为标准答案的 token 序列，长度 $L$）：

$$\log\pi_\theta(g\mid h_{i,\leq t})=\frac{1}{L}\sum_{j=1}^{L}\log\pi_\theta(g_j\mid h_{i,\leq t},g_{<j})$$

$$r_{i,t}^{IG}=\log\pi_\theta(g\mid h_{i,\leq t})-\log\pi_\theta(g\mid h_{i,\leq t-1}),\quad 1\leq t<T \quad(\text{stop-gradient})$$

代码实现（[ray_trainer.py:1629](../RL/verl/trainer/ppo/ray_trainer.py#L1629) `_compute_async_info_gain` + [prealigned_vectorized.py:240-300](../RL/scrl/llm_agent/prealigned_vectorized.py#L240-L300)）：

1. 构造伪回复：`"Now there's enough information to answer\n</think>\n<answer>\n{gt}\n</answer><|im_end|>"`，并定位 GT 部分的 token 区间；
2. 对每个保留下来的轮边界，把「截至该轮的真实上下文 + 伪回复」送入当前 actor，取 GT token 的平均 log-prob（复用 KV cache，分块 8192）；
3. `step 0`（只看到问题时）作为基线；之后每个边界的 IG = 当前平均 log-prob − 上一个**被计算过的**边界的平均 log-prob（`INFO_GAIN_TYPE=log_prob_diff`；另有 `prob_diff` 用概率差）；
4. **只在 visit 轮计算**（`IG_TOOL_FILTER="visit"`）：只有前一轮调用了 visit 的边界才保留。由于基线只在被计算的边界更新，一次 visit 得到的 IG 实际包含了自上次 visit 以来所有 search 带来的信息变化。这对应论文的 browse-aware assignment：IG 计算在 browse 轮上，并分配给该 browse 轮以及它之前、自上一次 browse 以来的 search 轮（分配是在优势计算阶段完成的，见 2.5）；
5. IG 值为 0 时写成 1e-10，用于标记“这里是一个奖励位置”。

把各轮奖励放到 token 上：[info_gain.py](../RL/verl/utils/reward_score/info_gain.py) `compute_score` 用 `<|im_end|>` 后紧跟 `<|im_start|>assistant` 这一特殊 token 锚点找到每轮的结束位置，把第 $t$ 轮 IG 写在该位置，结果奖励写在 `scores[-1]`（直接在原始 token id 上定位，避免 decode→re-encode 的 BPE 漂移）。

#### (3) 轮级格式惩罚

代码：[format_checker.py](../RL/verl/utils/reward_score/format_checker.py) `check_turn_format` + [ray_trainer.py:1937-1990](../RL/verl/trainer/ppo/ray_trainer.py#L1937-L1990)，`USE_FORMAT_PENALTY=true`、`FORMAT_PENALTY_SCALE=1.0`。

- 工具调用轮必须严格为 `<think>…</think><tool_call>{"name":…, "arguments":{…}}</tool_call>`：think 标签成对且在最前；不能出现 `<answer>`；恰好一个 tool_call；标签之间和结尾没有多余文本；tool_call 是合法 JSON，且含 `name` 和字典类型的 `arguments`；
- 回答轮必须严格为 `<think>…</think><answer>非空</answer>`，不能有 tool_call；
- 不合格时：工具调用轮的奖励（不论原来是 IG 还是没有奖励）替换为 −1；回答轮的结果奖励替换为 −1（通过 `answer_format_penalty` 覆盖 LLM judge 分数，见 [naive_batch.py:197-237](../RL/verl/workers/reward_manager/naive_batch.py#L197-L237)）。

论文公式：

$$\hat{r}_{i,t}=\begin{cases}r_{i,t}, & \text{第 } t \text{ 轮格式合法}\\ -\lambda_{\mathrm{fmt}}, & \text{否则}\end{cases},\qquad \lambda_{fmt}=1.0$$

### 2.5 优势计算：`grpo_info_gain`

代码：[RL/verl/trainer/ppo/info_gain_advantage.py](../RL/verl/trainer/ppo/info_gain_advantage.py)，`INFO_GAIN_NORM_MODE=scaled_separate`，`gamma=0.95`。

1. **区分两类奖励位置**：`f1_mask` 为每条轨迹最后一个有效 token（结果奖励）；`ig_mask` 为其它非零奖励位置（IG 或格式惩罚）。
2. **在组内分别做 z-score**（同一问题的 8 条 rollout 为一组）：
   - 结果奖励：$\tilde r^O=(\hat r^O-\mu^O)/\sigma^O$
   - 轮级奖励：$\tilde r^{IG}=(\hat r^{IG}-\mu^{IG})/\sigma^{IG}$
3. **IG 缩放**（防止稠密的 IG 信号压过结果奖励，或者被结果奖励淹没）：

$$M^{O}=\text{mean}|\tilde r^O|,\quad M^{IG}=\text{mean}|\tilde r^{IG}|,\quad s=\min\!\left(\frac{\max(M^{O},0.3)}{M^{IG}+\delta},\,10\right)$$

   只有 IG 乘以 $s$，结果奖励不变；之后再乘以 `IG_WEIGHT`（默认 1.0）。
4. **轮级折扣累积并广播到 token**（`_compute_turn_level_advantage`）：

$$\tilde R_{i,t}=\sum_{k=t}^{T_i}\gamma^{k-t}\bar r_{i,k},\qquad \gamma=0.95$$

   从后往前累加；第 $t$ 个奖励位置的回报 $\tilde R_{i,t}$ 会赋给**从上一个奖励位置之后到当前位置**的所有 token。因此一次 visit 之前的 search 轮也会拿到这次 visit 的回报，这就是 browse-aware assignment 的具体实现。

其它可选模式：`separate`（不缩放）、`joint`（两类奖励一起归一化）、`raw_ig`（IG 只除以 std 不减均值，并把结果奖励广播到每个 IG 位置，同时令 γ=0）。

### 2.6 策略更新

[RL/verl/workers/actor/dp_actor.py](../RL/verl/workers/actor/dp_actor.py)，配置来自 `train_igpo.sh` 和 `verl/trainer/config/actor/actor.yaml`：

| 项 | 值 |
|---|---|
| 目标 | PPO clipped surrogate，clip ε=0.2，`loss_agg_mode=token-mean`，ppo_epochs=1 |
| KL | `use_kl_loss=true`，`low_var_kl`，系数 0.001（不放进奖励） |
| 损失掩码 | `mask_tool_response=true`：只保留 `<|im_start|>assistant … <|im_end|>` 段内的 token（[dp_actor.py:906](../RL/verl/workers/actor/dp_actor.py#L906)），工具返回不参与梯度 |
| lr | 1e-6 |
| batch | 16 个问题 × 8 条 rollout，ppo_mini_batch_size=128，dynamic bsz |
| 长度 | `max_model_len=261000`，单轮生成 8192，Ulysses SP=8 |
| 显存 | actor/ref 参数与优化器 offload、activation offload |
| 其它 | 1 epoch，每 5 步保存一次；硬件为 16×A100（论文） |

论文未给出 RL 的 lr、步数、ε、β 的具体数值，以上取自仓库脚本。

### 2.7 RL 消融（论文）

| 设置 | BrowseComp | BrowseComp-ZH |
|---|---|---|
| SFT（重采样） | 26.8 | 35.7 |
| + RL（GRPO，只有结果奖励） | 25.3（−1.5） | 35.6（−0.1） |
| + RL（IGPO） | **29.1（+2.3）** | **37.7（+2.0）** |

结论：在超长程（上百轮）、数据量少（1K）的场景下，只有结果奖励的 GRPO 会出现组内回报相同导致的优势坍缩，几乎不涨；IGPO 的稠密轮级 IG 信号带来了稳定提升。论文没有单独消融格式惩罚、IG-Scale 和 browse-aware assignment 各自的贡献。

---

## 3. 关键开关速查（`RL/train_igpo.sh`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `USE_INFO_GAIN` | true | 开启后使用 `grpo_info_gain`，否则退化为普通 `grpo` |
| `INFO_GAIN_TYPE` | log_prob_diff | IG 用 log-prob 差还是 prob 差 |
| `INFO_GAIN_NORM_MODE` | scaled_separate | 优势归一化方式 |
| `IG_TOOL_FILTER` | visit | 只在这些工具轮上计算 IG（为空则所有轮都算） |
| `IG_COMPUTE_FREQ` | 1 | 每隔几轮计算一次 IG |
| `IG_WEIGHT` | 1.0 | IG 归一化后的额外权重 |
| `USE_FORMAT_PENALTY` / `FORMAT_PENALTY_SCALE` | true / 1.0 | 轮级格式惩罚 |
| `TRAIN_REWARD_TYPE` | llm | 结果奖励：`llm`（judge）或 `f1` |
| `MASK_TOOL_RESPONSE` | true | 工具返回不参与 loss |
| `USE_ROLLOUT_IG` | false | 实验性：在 vLLM rollout 时用 prompt_logprobs 计算 IG（快 10–15%，但与 FSDP 数值可能有差异） |
| `USE_ROBUST_ROLLOUT` | false | 遇到退化输出、解析错误、重复调用时重试该轮 |
