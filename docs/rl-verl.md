# RL 对 verl 框架的修改

> 本文说明 `RL/verl/` 相对于上游 [verl-project/verl](https://github.com/verl-project/verl) 改了哪些地方，以及它与 `SFT/verl/` 的关系。
> SFT 的对应文档见 [sft-verl.md](sft-verl.md)，版本演进见 [verl-history.md](verl-history.md)，RL 算法原理见 [overview.md §2](overview.md)。

---

## 1. 结论

- **基线版本相同**：`RL/verl` 与 `SFT/verl` 都基于上游提交 `a7b8675`（2025-09-15，`0.5.0.dev`）。
- **改动规模相差很大**：SFT 实际只改了 1 个文件中的 2 处；RL 修改或新增了 **41 个文件**，其中训练主循环 `ray_trainer.py` 改动约 2000 行。
- **两份 verl 各自独立修改**，不能混用：SFT 训练必须用 `SFT/verl`，RL 训练必须用 `RL/verl`。
- RL 的部分核心逻辑**不在 verl 中**，而是放在 `RL/scrl/llm_agent/` 下，通过配置注入 verl（见第 5 节）。

---

## 2. 如何确定基线

方法与 [sft-verl.md §2](sft-verl.md) 相同：把 `RL/verl/` 中每个文件的 git 哈希与上游各提交对比。

| | `SFT/verl` | `RL/verl` |
|---|---|---|
| 文件数 | 328 | 328 |
| 最佳匹配的上游提交 | `a7b8675` | **`a7b8675`**（与相邻的 `65170f9`、`44b919e` 并列，均为 287） |
| 与基线完全一致的文件 | 301 | 287 |
| 修改或新增的文件 | 27 | **41** |
| 删除的文件 | 0 | 0 |

> **注意**：`RL/verl/version/version` 写的是 **`0.1.0`**，而上游该提交是 `0.5.0.dev`（`SFT/verl` 也保持 `0.5.0.dev`）。这个版本号是手动改的，**不代表它是 verl 0.1.0**，不能用来判断版本。

---

## 3. 改动分类

改动量为与基线 diff 的增删行数之和；“新增”为新文件的行数。

### 3.1 IGPO 算法（信息增益奖励与优势）

| 文件 | 改动量 | 内容 |
|---|---|---|
| 新增 [trainer/ppo/info_gain_advantage.py](../RL/verl/trainer/ppo/info_gain_advantage.py) | 377 | 注册优势估计器 `grpo_info_gain`：在 GRPO 基础上区分最终答案奖励（F1/judge）和每轮信息增益奖励，支持多种归一化方式（`joint` / `separate` / `scaled_separate` / `raw_ig`）、课程权重和轮级折扣 |
| 新增 [utils/reward_score/info_gain.py](../RL/verl/utils/reward_score/info_gain.py) | 380 | 按 token 找出每一轮的边界，计算每轮的信息增益奖励和最终答案的 F1 |
| [trainer/config/algorithm.py](../RL/verl/trainer/config/algorithm.py) | 25 | 新增 IGPO 的全部配置项：`use_info_gain`、`info_gain_norm_mode`、`use_curriculum`、`use_kv_cache_ig`、`ig_tool_filter`、`ig_weight`、`ig_warmup_threshold`、`use_format_penalty`、`format_penalty_scale` 等 |
| [trainer/ppo/core_algos.py](../RL/verl/trainer/ppo/core_algos.py) | 114 | 新增另一个优势估计器 `mtgrpo`（多轮 GRPO，按轮折扣回报）。`train_igpo.sh` 默认使用 `grpo_info_gain`，关闭 IG 时使用 `grpo`，不使用 `mtgrpo` |

### 3.2 训练主循环

| 文件 | 改动量 | 内容 |
|---|---|---|
| [trainer/ppo/ray_trainer.py](../RL/verl/trainer/ppo/ray_trainer.py) | **2028** | 改动最大的文件，把 IGPO 串进 PPO 主循环，主要包括：<br>• `compute_advantage`（[:198](../RL/verl/trainer/ppo/ray_trainer.py#L198)）中分派 `mtgrpo`（[:265](../RL/verl/trainer/ppo/ray_trainer.py#L265)）和 `grpo_info_gain`（[:290](../RL/verl/trainer/ppo/ray_trainer.py#L290)）<br>• 用 KV cache 计算每轮信息增益（[:1636](../RL/verl/trainer/ppo/ray_trainer.py#L1636) 起，调用 `scrl.llm_agent.prealigned_vectorized`）<br>• 轮级格式惩罚：格式错误的轮次奖励设为 `-1 × format_penalty_scale`（[:1943](../RL/verl/trainer/ppo/ray_trainer.py#L1943) 起）<br>• IG 预热门控：最终答案得分的 EMA 达到阈值后才启用信息增益奖励，状态随 checkpoint 保存（[:2077](../RL/verl/trainer/ppo/ray_trainer.py#L2077) 起）<br>• `_dump_generations` 导出生成样本（[:471](../RL/verl/trainer/ppo/ray_trainer.py#L471)）<br>• 异步奖励数据集 `AsycRMDataset` 的分支（[:2634](../RL/verl/trainer/ppo/ray_trainer.py#L2634)） |
| [trainer/constants_ppo.py](../RL/verl/trainer/constants_ppo.py) | 42 | 把 `API_KEY`、`SERPER_KEY_ID`、`JINA_API_KEYS`、`JUDGE_*` 等环境变量显式传给每个 Ray worker。多机训练时远端 worker 不会继承驱动进程的环境变量，不传的话搜索和判分会因为缺少 key 而失败 |
| [trainer/ppo/metric_utils.py](../RL/verl/trainer/ppo/metric_utils.py) | 14 | 多轮相关指标，numpy 标量转为 Python 类型以便 JSON 序列化 |

### 3.3 Actor 与 Worker

| 文件 | 改动量 | 内容 |
|---|---|---|
| [workers/actor/dp_actor.py](../RL/verl/workers/actor/dp_actor.py) | 555 | 新增 `compute_ig_with_kv_cache`：先对整条轨迹前向一次得到 KV cache，再对每一轮截断 KV cache，只前向“标准答案”那几个 token，得到每轮的对数概率，从而算出信息增益。为避免 170K token 的轨迹显存爆炸，KV cache 分块构建；计算时临时关闭 Ulysses 序列并行；保证各 rank 的前向次数相同，避免 FSDP 的 ALLGATHER 死锁 |
| [workers/fsdp_workers.py](../RL/verl/workers/fsdp_workers.py) | 108 | 把 `compute_ig_with_kv_cache` 注册为可分发的 worker 方法；新增 `convert_to_rollout` / `convert_to_trainer`（在训练和推理模式间切换）、`generate_sequences_continue`（接着已有内容继续生成） |
| [workers/config/actor.py](../RL/verl/workers/config/actor.py) | 1 | 配置项补充 |
| [utils/ulysses.py](../RL/verl/utils/ulysses.py)、[utils/torch_functional.py](../RL/verl/utils/torch_functional.py) | 4 + 4 | 序列并行反向传播、数值计算的小修正 |
| [protocol.py](../RL/verl/protocol.py) | 22 | `DataProto` 的小改动，配合上面的数据传递 |

### 3.4 多轮 Rollout

| 文件 | 改动量 | 内容 |
|---|---|---|
| [experimental/agent_loop/agent_loop.py](../RL/verl/experimental/agent_loop/agent_loop.py) | 399 | 通用 agent loop 框架的增强：转发 `compute_prompt_logprobs` 到推理服务；记录 `num_turns`、`tool_timeout_count` 等指标；每个 worker 的 stdout 写入独立日志文件（`worker_stdout_logs/`），方便排查崩溃；队列和超时处理。**Deep Research 的具体逻辑 `DRAgentLoop` 不在这里**，见第 5 节 |
| [workers/rollout/vllm_rollout/vllm_async_server.py](../RL/verl/workers/rollout/vllm_rollout/vllm_async_server.py) | 80 | 异步推理服务：记录 prefill（TTFT）和 decode 耗时等 |
| [workers/rollout/vllm_rollout/vllm_rollout_spmd.py](../RL/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py) | 85 | 主要是注释掉上游对 `compute_logits` 的 monkey patch，以及调试输出 |
| [workers/rollout/async_server.py](../RL/verl/workers/rollout/async_server.py)、[workers/config/rollout.py](../RL/verl/workers/config/rollout.py) | 20 + 6 | 配套的服务和配置改动 |
| [utils/dataset/rl_dataset.py](../RL/verl/utils/dataset/rl_dataset.py) | 285 | 新增 `AsycRMDataset`：流式读取数据、跳过过期样本、异步回填奖励。只有设置 `data.custom_train_cls.name=AsycRMDataset` 时才启用 |

### 3.5 奖励

| 文件 | 改动量 | 内容 |
|---|---|---|
| 新增 [utils/reward_score/llm_judge.py](../RL/verl/utils/reward_score/llm_judge.py) | 222 | 调用 LLM 判断最终答案是否正确（`train_reward_type=llm`） |
| 新增 [utils/reward_score/format_and_f1.py](../RL/verl/utils/reward_score/format_and_f1.py) | 128 | 格式检查 + F1 奖励（`train_reward_type=f1`） |
| 新增 [utils/reward_score/format_checker.py](../RL/verl/utils/reward_score/format_checker.py) | 116 | `check_turn_format`：检查单轮的 `<think>` / `<tool_call>` / `<answer>` 格式，供格式惩罚使用 |
| 新增 [workers/reward_manager/naive_batch.py](../RL/verl/workers/reward_manager/naive_batch.py) | 328 | 批量调用奖励函数，并叠加最终答案的格式惩罚 |
| [workers/reward_manager/naive.py](../RL/verl/workers/reward_manager/naive.py) | 51 | 支持信息增益：解码时保留特殊 token（用来识别轮次边界），返回 token 级奖励 |
| [trainer/ppo/reward.py](../RL/verl/trainer/ppo/reward.py)、[utils/reward_score/\_\_init\_\_.py](../RL/verl/utils/reward_score/__init__.py)、[workers/reward_manager/\_\_init\_\_.py](../RL/verl/workers/reward_manager/__init__.py) | 54 + 38 + 2 | `compute_reward` / `compute_reward_async` 增加 `val_type` 参数；注册新的奖励函数和奖励管理器 |

### 3.6 其他

| 文件 | 内容 |
|---|---|
| `utils/flops_counter.py`、`utils/profiler/performance.py` | 与 SFT 相同：PPU-ZW810E 的算力数值、rank 0 打印计时日志 |
| `utils/reward_score/prime_code/testing_util.py` | 2 行小改动 |
| `version/version` | `0.5.0.dev` → `0.1.0`（见第 2 节） |
| 若干 yaml（`ppo_trainer.yaml`、`actor.yaml`、`rollout.yaml`、`reward_model/*.yaml`、`data/legacy_data.yaml` 及生成的配置） | 为上述新功能增加配置项 |

---

## 4. 与 `SFT/verl` 的关系

两份 verl 是从同一个基线**分叉后各自修改**的，不是一份代码复制成两份。

| 情况 | 文件 |
|---|---|
| 只在 RL 中有 | `trainer/ppo/info_gain_advantage.py`、`utils/reward_score/info_gain.py`、`utils/reward_score/format_checker.py` —— IGPO 只存在于 RL |
| 只在 SFT 中有 | `utils/reward_score/stock_judge.py`、`utils/reward_score/fact_test.py`、`workers/reward_manager/group_async_batch.py` —— 与 Deep Research 无关，SFT 训练也不使用，像是其他项目遗留的文件 |
| 两边都改但改法不同 | `ray_trainer.py`、`core_algos.py`、`agent_loop.py`、`rl_dataset.py` 等 —— SFT 中的 RL 代码是较早的中间版本 |
| SFT 改了、RL 没改 | `trainer/fsdp_sft_trainer.py` —— RL 中与上游完全一致；SFT 中关闭了 sampler 打乱、修正了学习率调度总步数（见 [sft-verl.md §4](sft-verl.md)） |

---

## 5. verl 之外的 RL 代码

RL 的部分核心逻辑放在 [RL/scrl/llm_agent/](../RL/scrl/llm_agent/) 下，不属于 verl：

| 文件 | 作用 | 如何接入 verl |
|---|---|---|
| [dr_agent_loop.py](../RL/scrl/llm_agent/dr_agent_loop.py) | `DRAgentLoop`：Deep Research 的多轮工具调用 rollout | 通过 [configs/dr_agent_loop.yaml](../RL/configs/dr_agent_loop.yaml) 注册为 `dr_agent`；`train_igpo.sh` 在 `USE_ASYNC_ROLLOUT=true` 时用 `actor_rollout_ref.rollout.agent.agent_loop_config_path` 指向它 |
| [prealigned_vectorized.py](../RL/scrl/llm_agent/prealigned_vectorized.py) | 用 KV cache 批量计算每轮信息增益 | 在 `ray_trainer.py` 中直接 import（[:1641](../RL/verl/trainer/ppo/ray_trainer.py#L1641)） |
| `generation.py`、`tensor_helper.py` | 生成和张量处理的辅助函数 | 被上面的模块使用 |

工具服务（搜索、访问网页、摘要）在 [RL/tool_server/](../RL/tool_server/) 下，同样不在 verl 中。

---

## 6. 对升级的影响

| | SFT | RL |
|---|---|---|
| 改动的 verl 文件 | 1 个（2 处） | 41 个 |
| 依赖的上游接口 | `fsdp_sft_trainer.py`（v0.7.1 删除） | `ray_trainer.py`、FSDP worker、SPMD/async rollout、reward manager、agent loop |
| 升级难度 | 低：把两处修改迁移到新的 `sft_trainer.py` | 高：v0.7 移除 SPMD rollout，v0.8 废弃旧 worker，v0.9 默认启用 V1 trainer，基本等于重新移植 |

各版本不兼容改动的细节见 [verl-history.md §4](verl-history.md)。
