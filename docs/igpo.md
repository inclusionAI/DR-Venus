# IGPO：基于信息增益的策略优化

> 本文解释 DR-Venus RL 阶段使用的 IGPO 算法：要解决的问题、算法思路、一个完整的数值例子，以及它在本仓库 verl 中的实现链路和工程细节。
> 相关文档：RL 整体流程见 [overview.md §2](overview.md)，RL 对 verl 的改动见 [rl-verl.md](rl-verl.md)。

---

## 1. 要解决的问题

DR-Venus 的 RL 场景有三个特点：

- **轨迹很长**：一条轨迹动辄几十到上百轮工具调用，上下文超过 200K token。
- **奖励很稀疏**：只有最后的 `<answer>` 由 LLM judge 判对或判错，得到 0 或 1。
- **数据很少**：只有 1K 个问题，每个问题 rollout 8 条。

标准 GRPO 在这里会遇到两个问题。

**问题 1：优势坍缩。** GRPO 的优势是“组内奖励的 z-score”。如果同一个问题的 8 条 rollout 全部答对或全部答错（难题全错、简单题全对，这很常见），组内方差为 0，所有 token 的优势都是 0，这一组对梯度毫无贡献。代码中专门统计了这个比例：`igpo/all_same_outcome_ratio`（见 [info_gain_advantage.py](../RL/verl/trainer/ppo/info_gain_advantage.py) 末尾的指标部分）。

**问题 2：信用分配太粗。** 即使组内有差异，GRPO 也会把同一个优势值广播到整条轨迹的所有 token。一条答对的轨迹可能有 100 轮操作，其中只有 3 次 visit 真正找到了关键证据，另外 97 轮是无效搜索，但它们得到的奖励完全一样。

论文的消融结果（见 [overview.md §2.7](overview.md)）：在 SFT 模型上直接做只有结果奖励的 GRPO，BrowseComp **下降** 1.5 分；换成 IGPO 则**提升** 2.3 分。

---

## 2. 核心思路

IGPO 为每一轮工具调用计算一个**稠密的过程奖励**：把标准答案 $g$ 当成“伪回复”，问当前模型——**看到这一轮获得的新信息之后，你直接说出标准答案的概率提高了多少？**

**第一步：标准答案的平均对数概率。** $h_{\le t}$ 是截至第 $t$ 轮的上下文，包括问题、前 $t$ 轮的思考和工具调用，以及工具返回的结果：

$$\log\pi_\theta(g\mid h_{\le t})=\frac{1}{L}\sum_{j=1}^{L}\log\pi_\theta(g_j\mid h_{\le t},g_{<j})$$

**第二步：信息增益。**

$$r^{IG}_t=\log\pi_\theta(g\mid h_{\le t})-\log\pi_\theta(g\mid h_{\le t-1})$$

- 为正：这一轮的信息让模型更确信正确答案。
- 接近 0：这一轮没有带来有用信息。
- 为负：这一轮把模型带偏了。

$\pi_\theta$ 是**当前的 actor 模型本身**，不需要另外训练奖励模型。计算在 `no_grad` 下进行，IG 只是一个奖励数值，不回传梯度。

**第三步：与结果奖励组合，计算优势。** 结果奖励和 IG 奖励**分别**在组内归一化，把 IG 缩放到与结果奖励相同的量级，再按轮做折扣累积（$\gamma=0.95$），最后广播到每一轮的 token 上。

效果：

- 即使 8 条 rollout 全部答错，不同轨迹的 IG 仍有差异（有的找到了相关证据，有的没有），这一组依然能产生有效梯度，缓解问题 1。
- 每一轮得到的优势不再相同，缓解问题 2。

---

## 3. 完整示例

**问题**：“DR-Venus 论文是哪一年发布的？”，标准答案 $g$ = `2026`。

某一条 rollout 的过程：

| 轮 | 动作 | 格式 |
|---|---|---|
| 第 1 轮 | search "DR-Venus" | ✅ |
| 第 2 轮 | visit arxiv 页面 | ✅ |
| 第 3 轮 | search "DR-Venus release" | ❌ 格式错误 |
| 第 4 轮 | visit 另一个页面 | ✅ |
| 第 5 轮 | `<answer>2026</answer>` | ✅，judge 判对，奖励 = 1 |

### 3.1 记录轮边界

Rollout 时，`DRAgentLoop` 为每一轮记录一个边界 `turn_boundaries[k]`：

- `token_len`：边界 k 处的上下文长度，即已经执行完 k 次工具调用、即将生成第 k+1 轮时的位置。
- `tool_name`：在这个边界之后生成的那一轮调用的工具。

边界 0 只包含问题本身，作为基线。

### 3.2 只在 visit 轮计算 IG

默认 `IG_TOOL_FILTER="visit"`：边界 k 只有在它前面那一轮（第 k 轮）调用的是 visit 时才保留。因此本例只计算边界 0、2、4。

假设算出的平均对数概率为：

| 边界 | 上下文 | $\log\pi(g\mid\cdot)$ | IG |
|---|---|---|---|
| 0 | 只有问题 | −3.0 | 基线 |
| 2 | 执行完第 1、2 轮 | −1.8 | −1.8 − (−3.0) = **+1.2** |
| 4 | 执行完第 1–4 轮 | −0.4 | −0.4 − (−1.8) = **+1.4** |

基线只在**实际计算过的边界**上更新，所以边界 2 的 IG 包含了第 1 轮 search 和第 2 轮 visit 带来的全部信息变化。这对应论文中的 *browse-aware assignment*：search 通常只返回摘要，真正的信息靠 visit 读到，所以信息增益记在 visit 轮上。

### 3.3 展开成每轮一个值，并叠加格式惩罚

把稀疏结果展开，每个非基线轮次一个值，未计算的轮次填 `None`：

```
ig_rewards = [None, 1.2, None, 1.4]     # 分别对应第 1–4 轮
```

格式检查（`check_turn_format`）发现第 3 轮格式错误，将其替换为 −1：

```
ig_rewards = [None, 1.2, -1.0, 1.4]
```

如果是最后的回答轮格式错误，则用 −1 覆盖结果奖励。

### 3.4 把奖励写到 token 上

`compute_score`（[info_gain.py:244](../RL/verl/utils/reward_score/info_gain.py#L244)）在原始 token 序列中寻找“`<|im_end|>` 后紧跟 `<|im_start|>assistant`”的位置，这就是**每一轮工具返回的结尾**。第 i 轮的奖励写在第 i 个这样的位置上，结果奖励写在最后一个 token 上：

```
token:  [第1轮 search | 返回] [第2轮 visit | 返回] [第3轮 search | 返回] [第4轮 visit | 返回] [第5轮 answer]
reward:              0                 1.2                -1.0                 1.4               1.0
                    (None 跳过)        p2                  p3                  p4                last
```

- 值为 `None` 的位置不写入。
- IG 恰好为 0 的位置写成 `1e-10`，这样它仍会被识别为“奖励位置”，不会被当成“没有奖励”而丢掉。

### 3.5 计算优势（`grpo_info_gain`，默认 `scaled_separate` 模式）

**(a) 区分两类奖励位置。** `f1_mask` 是每条轨迹的最后一个 token（结果奖励）；`ig_mask` 是其他所有非零奖励位置（IG 和格式惩罚）。

**(b) 组内分别做 z-score。** 结果奖励用同组 8 个最终得分计算均值和方差；IG 用同组 8 条轨迹中所有 IG 位置的值计算均值和方差。

**(c) 缩放 IG：**

$$s=\min\left(\frac{\max(\text{mean}|\tilde r^O|,\,0.3)}{\text{mean}|\tilde r^{IG}|},\ 10\right)$$

只有 IG 乘以 $s$，使两类信号量级相当，避免 IG 压过结果奖励，也避免 IG 被结果奖励淹没。之后再乘以 `IG_WEIGHT`（默认 1.0）。

**(d) 按轮折扣累积，从后往前算。** 设归一化后的值为 $\bar r_{2}, \bar r_{3}, \bar r_{4}, \bar r_O$：

$$A_{last}=\bar r_O,\quad A_{p4}=\bar r_4+0.95A_{last},\quad A_{p3}=\bar r_3+0.95A_{p4},\quad A_{p2}=\bar r_2+0.95A_{p3}$$

**(e) 广播到 token。** 每个奖励位置的优势，赋给从**上一个奖励位置之后到当前位置**的所有 token：

| token 区间 | 优势 |
|---|---|
| `[0, p2]`：第 1 轮 search + 第 2 轮 visit | $A_{p2}$ |
| `(p2, p3]`：第 3 轮（格式错误） | $A_{p3}$，包含 −1 带来的负值 |
| `(p3, p4]`：第 4 轮 visit | $A_{p4}$ |
| `(p4, last]`：回答轮 | $A_{last}$ |

第 1 轮的 search 分享了第 2 轮 visit 的回报——这就是“信息增益记在 visit 上，信用分给它之前的 search”在代码中的具体实现（`_compute_turn_level_advantage`）。

### 3.6 更新策略

最后做 PPO 裁剪更新。`mask_tool_response=true` 把工具返回部分的 token 从损失中去掉，只有模型自己生成的 token 参与梯度计算。

---

## 4. 在 verl 中的实现链路

IGPO 没有新写 trainer，而是插进了 verl 原有的 PPO 主循环 `RayPPOTrainer.fit()`。每个训练 step 的流程：

```
fit()  ray_trainer.py
 │
 ├─① Rollout   async_rollout_manager.generate_sequences()
 │             └─ DRAgentLoop (scrl/llm_agent/dr_agent_loop.py)
 │                 产出 input_ids、messages、turn_boundaries[{step, token_len, msg_count, tool_name}]
 │
 ├─② IG 预热门控（可选）    ray_trainer.py:2216
 │
 ├─③ 计算信息增益   _compute_async_info_gain()   ray_trainer.py:1629
 │   ├ 构造伪回复：PREFIX + gt + SUFFIX，用 offset_mapping 定位 gt 的 token 区间
 │   ├ 按 IG_TOOL_FILTER 和 IG_COMPUTE_FREQ 筛选边界
 │   ├ 直接复用 agent 生成时的 input_ids（避免 BPE 漂移）
 │   ├ scrl/llm_agent/prealigned_vectorized.py:486  compute_ig_with_kv_cache
 │   │   └ actor_rollout_wg.compute_ig_with_kv_cache  →  fsdp_workers.py（分发到各 rank）
 │   │       └ dp_actor.py:433  _compute_ig_with_kv_cache_impl（GPU 上计算每轮 log-prob）
 │   │   ← 返回每轮 log-prob，在 driver 上做差分得到 IG
 │   ├ 稀疏结果展开成每轮一个值（None 补位）
 │   └ 格式惩罚：工具调用轮 → 覆盖 IG；回答轮 → answer_format_penalty
 │   写入 batch.non_tensor_batch['info_gain_rewards' / 'answer_format_penalty']
 │
 ├─④ response_mask 全部设为 1      ray_trainer.py:2600
 │   （奖励在工具返回的结尾位置，如果这里就 mask 掉工具返回，IG 会全部丢失）
 │
 ├─⑤ 计算奖励   compute_reward(info_gain_rewards=…)        ray_trainer.py:2650
 │   └ NaiveBatchRewardManager (naive_batch.py)
 │       ├ LLM judge 得到结果奖励；answer_format_penalty 覆盖结果奖励
 │       └ info_gain.compute_score：把 IG 和结果奖励写到 token 位置 → token_level_rewards
 │
 ├─⑥ 计算 old_log_prob 和 ref_log_prob（原生 verl 流程）
 │
 ├─⑦ 计算优势   compute_advantage(adv_estimator="grpo_info_gain", config=algorithm)  ray_trainer.py:2808
 │   └ 分派到 :290 → info_gain_advantage.compute_grpo_info_gain_advantage
 │       分别归一化 → IG 缩放 → 按轮折扣 → 广播到 token；输出 igpo/* 指标
 │
 └─⑧ update_actor   dp_actor.update_policy
     └ mask_tool_response：process_response_mask 去掉工具返回的 token → PPO loss
```

### 4.1 用到的 verl 扩展点

| verl 扩展点 | IGPO 的用法 |
|---|---|
| `@register_adv_est` 注册优势估计器 | 注册 `"grpo_info_gain"`（[info_gain_advantage.py:82](../RL/verl/trainer/ppo/info_gain_advantage.py#L82)）；`ray_trainer.py:2107` import 该模块以触发注册 |
| `AlgoConfig` 数据类 | 新增 IGPO 全部配置项（[algorithm.py](../RL/verl/trainer/config/algorithm.py)），可通过 `algorithm.xxx=` 在命令行传参 |
| `@register(dispatch_mode=...)` 定义 worker 方法 | 在 FSDP worker 上新增 `compute_ig_with_kv_cache`，由 driver 通过 RPC 分发到各 rank |
| reward manager 与 `compute_score` | 新增 `naive_batch` 管理器，`compute_score` 返回 token 级奖励列表而非单个分数 |
| agent loop 配置 | 通过 `configs/dr_agent_loop.yaml` 注入 `DRAgentLoop`，由它记录轮边界 |

### 4.2 关键代码位置

| 功能 | 位置 |
|---|---|
| 轮边界记录 | [scrl/llm_agent/dr_agent_loop.py](../RL/scrl/llm_agent/dr_agent_loop.py) |
| IG 计算入口（driver 侧） | [ray_trainer.py:1629](../RL/verl/trainer/ppo/ray_trainer.py#L1629) `_compute_async_info_gain` |
| IG 差分 | [prealigned_vectorized.py:486](../RL/scrl/llm_agent/prealigned_vectorized.py#L486) `compute_ig_with_kv_cache` |
| KV cache 前向（GPU 侧） | [dp_actor.py:433](../RL/verl/workers/actor/dp_actor.py#L433) `_compute_ig_with_kv_cache_impl` |
| 格式检查 | [format_checker.py](../RL/verl/utils/reward_score/format_checker.py) `check_turn_format`；应用于 [ray_trainer.py:1943](../RL/verl/trainer/ppo/ray_trainer.py#L1943) 起 |
| 奖励写到 token | [info_gain.py:101](../RL/verl/utils/reward_score/info_gain.py#L101) `_find_turn_boundaries_by_tokens`、[:244](../RL/verl/utils/reward_score/info_gain.py#L244) `compute_score` |
| 优势计算 | [info_gain_advantage.py](../RL/verl/trainer/ppo/info_gain_advantage.py) `compute_grpo_info_gain_advantage`、`_compute_turn_level_advantage` |
| 损失掩码 | [dp_actor.py](../RL/verl/workers/actor/dp_actor.py) `process_response_mask` / `update_policy` |

### 4.3 Rollout 如何接入 `DRAgentLoop`

链路图第 ① 步 `async_rollout_manager.generate_sequences() → DRAgentLoop` 大部分沿用 verl 原有的机制，项目只在 4 个地方做了改动或接入。

#### 上游原有的机制（未修改）

verl 0.5 的 `experimental/agent_loop` 已经提供了一套可插拔的多轮 agent rollout：

```
RayPPOTrainer.__init__
  └ rollout.mode == "async" 时创建 AgentLoopManager        ray_trainer.py:909-915
fit()
  └ async_rollout_manager.generate_sequences(gen_batch)
      └ AgentLoopManager → 把 batch 分片给多个 AgentLoopWorker（Ray actor）
          └ AgentLoopWorker.generate_sequences                 agent_loop.py:457
              ├ 按每条样本的 non_tensor_batch["agent_name"] 在注册表中查找
              │   _agent_loop_registry[agent_name]                agent_loop.py:704-709
              ├ hydra.utils.instantiate(...) 创建对应的 AgentLoop 实例
              ├ await agent_loop.run(...) → 返回 AgentLoopOutput
              └ _postprocess：把 AgentLoopOutput.extra_fields 中的每个 key
                  原样放进 non_tensor_batch                      agent_loop.py:909-914
```

其中三个扩展点是上游原本就有的：

| 扩展点 | 位置 | 作用 |
|---|---|---|
| `@register("名字")` 装饰器 | [agent_loop.py:270](../RL/verl/experimental/agent_loop/agent_loop.py#L270) | 把一个 `AgentLoopBase` 子类注册到 `_agent_loop_registry` |
| `rollout.agent.agent_loop_config_path` | [agent_loop.py:417-421](../RL/verl/experimental/agent_loop/agent_loop.py#L417-L421) | 从 yaml 读取 `{name, _target_}`，把外部的类注册进来，不需要修改 verl 源码 |
| `extra_fields` 透传 | [agent_loop.py:909-914](../RL/verl/experimental/agent_loop/agent_loop.py#L909-L914) | agent 返回的任何额外字段都会出现在 `gen_batch_output.non_tensor_batch` 中 |

因此 `DRAgentLoop` 不需要修改 verl 就能接入，它返回的 `turn_boundaries`、`messages` 也是通过原有的透传机制到达 trainer 的。

#### 改动 ①：新增 `DRAgentLoop` 并注册（不在 verl 中）

[RL/scrl/llm_agent/dr_agent_loop.py:164](../RL/scrl/llm_agent/dr_agent_loop.py#L164)：

```python
@register("dr_agent")
class DRAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput: ...
```

`run()` 实现 Deep Research 的多轮循环：生成 → 解析 `<tool_call>` → 调用 search/visit → 追加 `<tool_response>` → 下一轮。最后通过 `extra_fields` 返回 IGPO 需要的信息（[:790-819](../RL/scrl/llm_agent/dr_agent_loop.py#L790-L819)）：

```python
_extra = {"turn_boundaries": turn_boundaries, "messages": messages, "cutoff": _was_cutoff}
# 打开 rollout IG 时还有 "rollout_ig_rewards"
return AgentLoopOutput(prompt_ids=..., response_ids=..., response_mask=...,
                       num_turns=..., extra_fields=_extra)
```

`@register` 只在模块被 import 时才生效，而 verl 不会主动 import `scrl`，所以还需要配置文件 [RL/configs/dr_agent_loop.yaml](../RL/configs/dr_agent_loop.yaml)：

```yaml
- name: dr_agent
  _target_: scrl.llm_agent.dr_agent_loop.DRAgentLoop
```

#### 改动 ②：启动脚本打开 async 模式并指向该配置

[RL/train_igpo.sh:203-209](../RL/train_igpo.sh#L203-L209)，在 `USE_ASYNC_ROLLOUT=true`（默认）时传入：

```bash
actor_rollout_ref.rollout.mode=async                                   # 让 trainer 创建 AgentLoopManager
actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/dr_agent_loop.yaml
actor_rollout_ref.rollout.agent.num_workers=${ASYNC_NUM_WORKERS}
+actor_rollout_ref.rollout.agent.completion_cutoff=${ASYNC_COMPLETION_CUTOFF}
```

#### 改动 ③：trainer 中写死 `agent_name` 和重复次数（修改 verl）

[ray_trainer.py:2146-2152](../RL/verl/trainer/ppo/ray_trainer.py#L2146-L2152)：

```python
if self.async_rollout_mode:
    _async_n = self.config.agent_grpo.n
    gen_batch = gen_batch.repeat(repeat_times=_async_n, interleave=True)
    gen_batch.non_tensor_batch["agent_name"] = np.array(["dr_agent"] * len(gen_batch), dtype=object)
else:
    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
```

与上游的区别：

- **上游**：`agent_name` 来自数据集中的一列，没有这一列时 `AgentLoopWorker` 默认使用 `single_turn_agent`（[agent_loop.py:492-493](../RL/verl/experimental/agent_loop/agent_loop.py#L492-L493)）；重复次数使用 `rollout.n`。
- **本项目**：直接把所有样本的 `agent_name` 写死为 `"dr_agent"`，数据集不需要 `agent_name` 列；重复次数改用新增的配置项 `agent_grpo.n`（[ppo_trainer.yaml:374](../RL/verl/trainer/config/ppo_trainer.yaml#L374)，脚本中设为 8）。

`generate_sequences` 返回后，trainer 还会把整数 tensor 转为 long、根据 `__num_turns__` 构造 `agent_grpo_idx_deepthink`（[:2196-2214](../RL/verl/trainer/ppo/ray_trainer.py#L2196-L2214)），然后进入 IG 预热门控和信息增益计算。这些都是新增的代码。

#### 改动 ④：增强通用的 `agent_loop.py`（修改 verl）

[RL/verl/experimental/agent_loop/agent_loop.py](../RL/verl/experimental/agent_loop/agent_loop.py) 相对上游改了约 400 行，都是通用框架层的增强，不包含 Deep Research 的业务逻辑：

| 改动 | 位置 | 作用 |
|---|---|---|
| 转发 `compute_prompt_logprobs` | [:113-120](../RL/verl/experimental/agent_loop/agent_loop.py#L113-L120) | 让 agent 可以向 vLLM 请求 prompt 的 log-prob，供实验性的 rollout IG（`USE_ROLLOUT_IG`）使用 |
| `tool_timeout_count` 指标 | [:140](../RL/verl/experimental/agent_loop/agent_loop.py#L140)、[:1194](../RL/verl/experimental/agent_loop/agent_loop.py#L1194) | 统计工具调用超时次数 |
| `completion_cutoff` 截断拖尾 | [:512-534](../RL/verl/experimental/agent_loop/agent_loop.py#L512-L534) | 一定比例（脚本中为 90%）的样本完成后，通知剩余样本强制给出答案，避免整个 batch 被几条超长轨迹拖住 |
| 失败时返回占位输出 | [:601-654](../RL/verl/experimental/agent_loop/agent_loop.py#L601-L654) | 单条 agent 抛出异常时，返回奖励为 0、`turn_boundaries` 为空的占位输出，不让整个 step 崩溃；IG 和格式惩罚会跳过这类样本 |
| 每个 worker 单独的日志 | [:447](../RL/verl/experimental/agent_loop/agent_loop.py#L447) 附近 | stdout 写入 `worker_stdout_logs/worker_<pid>.log`（原来丢弃到 /dev/null），便于排查崩溃 |
| 完成进度统计 | [:672-677](../RL/verl/experimental/agent_loop/agent_loop.py#L672-L677) | 打印平均轮数、最大轮数和被截断的样本数 |

#### 小结

| 层次 | 文件 | 是否修改 verl |
|---|---|---|
| Deep Research 业务逻辑 | `scrl/llm_agent/dr_agent_loop.py`（`@register("dr_agent")`） | 否，外部插件 |
| 注册配置 | `configs/dr_agent_loop.yaml` | 否 |
| 启动参数 | `train_igpo.sh`：`rollout.mode=async`、`agent_loop_config_path` 等 | 否 |
| 选择 agent | `ray_trainer.py:2146-2152`：写死 `agent_name="dr_agent"`，按 `agent_grpo.n` 重复 | **是** |
| 通用框架增强 | `agent_loop.py`：截断拖尾、容错、日志、prompt logprobs | **是** |
| 调度、实例化、`extra_fields` 透传 | `AgentLoopManager` / `AgentLoopWorker` 主体 | 沿用上游 |

DRAgentLoop 本身按 verl 的插件方式写在外部；修改 verl 主要是为了让每条样本都走 `dr_agent`，以及让上百轮的长轨迹 rollout 在稳定性和效率上撑得住。

---

## 5. 工程细节

### 5.1 用 KV cache 计算 IG

最直接的做法是对每个边界把“截至该边界的上下文 + 伪回复”完整前向一次。但一条 100 轮、170K token 的轨迹，这需要 100 次长序列前向，根本算不动。实际分两个阶段（[dp_actor.py:433](../RL/verl/workers/actor/dp_actor.py#L433)）：

- **阶段 1**：对整条轨迹前向一次得到 KV cache。为控制显存按块构建，代码默认每块 4096 token，`train_igpo.sh` 设为 `IGPO_PHASE1_CHUNK_SIZE=8192`。
- **阶段 2**：对每个边界，把 KV cache 截断到该边界，再只前向伪回复的几十个 token，取出标准答案部分的 log-prob。

总计算量从“每个边界一次长序列前向”降为“一次长序列前向 + 每个边界一次很短的前向”。

另外两点：

- 计算时**临时关闭 Ulysses 序列并行**：它的 all-to-all 会打乱 position_ids，导致 flash attention 崩溃。
- **保证每个 rank 的前向次数相同**，不足的 rank 用空前向补齐。FSDP 每次前向都要 ALLGATHER，如果有 rank 先结束，其余 rank 会一直等待直到 NCCL 超时，形成死锁。

### 5.2 伪回复

伪回复的格式是：

```
Now there's enough information to answer\n</think>\n<answer>\n{gt}\n</answer><|im_end|>
```

只取其中 `{gt}` 部分 token 的 log-prob 求平均，前缀和后缀只用于把模型带到“即将给出答案”的状态。

### 5.3 避免 BPE 漂移

文本“解码 → 重新编码”后，token 边界可能和原来不一致。因此：

- 优先直接使用 agent 生成时的原始 `input_ids` 和 agent 记录的 `token_len`。
- 确定奖励位置时，只在原始 token 序列中查找 `<|im_end|>`、`<|im_start|>` 这类不会被拆分的特殊 token。
- 代码带有 crosscheck 和 verify 两套校验，可用环境变量 `IGPO_CROSSCHECK`、`IGPO_VERIFY_*` 打开。

### 5.4 `response_mask` 的两个阶段

奖励位置在工具返回的结尾，所以**计算优势时** `response_mask` 必须全部为 1。如果此时该位置被 mask 掉，代码会直接抛出 `RuntimeError`，而不是让 IGPO 悄悄退化成普通 GRPO。到 **`update_actor` 时**，才把工具返回的 token 从损失中去掉。

### 5.5 归一化模式

| `INFO_GAIN_NORM_MODE` | 做法 |
|---|---|
| `scaled_separate`（默认） | 结果奖励和 IG 分别 z-score，再把 IG 缩放到与结果奖励相同量级 |
| `separate` | 分别 z-score，不缩放 |
| `joint` | 两类奖励放在一起 z-score |
| `raw_ig` | IG 只除以标准差不减均值；把结果奖励加到每个 IG 位置；强制 $\gamma=0$ |

---

## 6. 配置开关

`train_igpo.sh` 中的相关配置：

| 变量 | 默认 | 作用 |
|---|---|---|
| `USE_INFO_GAIN` | true | 总开关。关闭后 `ADV_ESTIMATOR=grpo`，即普通 GRPO |
| `INFO_GAIN_TYPE` | `log_prob_diff` | 用对数概率差作为 IG；`prob_diff` 用概率差 |
| `INFO_GAIN_NORM_MODE` | `scaled_separate` | 见 §5.5 |
| `IG_TOOL_FILTER` | `visit` | 只在 visit 轮计算 IG；留空表示每轮都计算 |
| `IG_COMPUTE_FREQ` | 1 | 每隔几轮计算一次 IG |
| `IG_WEIGHT` | 1.0 | 归一化后 IG 的额外倍数 |
| `USE_FORMAT_PENALTY` / `FORMAT_PENALTY_SCALE` | true / 1.0 | 轮级格式惩罚 |
| `algorithm.gamma` | 0.95 | 按轮折扣系数 |
| `agent_grpo.n` | 8 | 每个问题 rollout 条数，即 GRPO 组大小 |
| `USE_ROLLOUT_IG` | false | 实验性：rollout 时直接用 vLLM 的 prompt_logprobs 计算 IG，省去训练端 KV cache 这一步 |

`AlgoConfig` 中还有几个默认关闭的可选功能：

| 配置项 | 作用 |
|---|---|
| `use_curriculum` 及 `curriculum_*` | 结果奖励和 IG 的权重随训练进度线性变化 |
| `ig_warmup_threshold` / `ig_warmup_ema_alpha` | 结果得分的 EMA 达到阈值后才启用 IG，启用后不再关闭；状态随 checkpoint 保存 |
| `adaptive_gamma_min_weight` | 按轨迹长度调整 $\gamma$，保证早期轮次的信号不会衰减得太小 |

训练时可在日志中关注 `igpo/*` 指标，例如 `igpo/ig_active_ratio`（有 IG 的样本比例）、`igpo/ig_scale`（IG 缩放系数）、`igpo/all_same_outcome_ratio`（结果奖励完全相同的组的比例）。

---

## 7. 小结

IGPO 把“当前模型说出标准答案的概率”当作一把尺子，衡量每一次 visit 带来的进展，从而把稀疏的 0/1 结果奖励补充成每一轮的过程奖励。实现上，它通过 verl 的优势估计器注册、worker 方法分发和 reward manager 三个扩展点接入原有的 PPO 主循环；最主要的工程工作，是用 KV cache 让 170K token 的长轨迹也能逐轮计算 IG。
