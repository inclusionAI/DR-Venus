# SFT 对 verl 框架的修改

> 本文说明 `SFT/verl/` 相对于上游 [volcengine/verl](https://github.com/volcengine/verl) 改了哪些地方，以及这些改动对 SFT 训练的影响。
> 脚本和参数见 [sft-script.md](sft-script.md)，数据清洗见 [sft-data.md](sft-data.md)。

---

## 1. 结论

这个项目为了做 SFT，**对 verl 框架本身改得很少**：和 SFT 有关的实际只有 `fsdp_sft_trainer.py` 一个文件，改了两处；另外有两处与训练逻辑无关的小改动。

SFT 能够运行，主要靠的是**把数据整理成 verl 原生多轮 SFT 能直接读取的格式**，再配合 shell 脚本传参，而不是修改框架。整体分三层：

| 层 | 内容 | 工作量 |
|---|---|---|
| 数据层 | `prepare_trajectories.py` 把轨迹整理成 verl 原生支持的 `messages` 格式，并负责打乱数据 | 主要工作 |
| 配置层 | `train_sft.sh` / `sft_shells/run.sh` 把论文超参数翻译成 Hydra 参数 | 中等 |
| 框架层 | 关闭 sampler 打乱，修正学习率调度的总步数 | 很少 |

---

## 2. 如何确定改动

仓库中的 `SFT/verl/` 只有一次提交，没有修改历史可查。确定改动的方法是：把它与上游 verl 的提交历史逐个对比文件的 git 哈希，找到最接近的基线。

- **基线提交**：`a7b8675`（2025-09-15，“[rollout] fix: make agent loop reward worker thread-safe”）
- **匹配情况**：328 个文件中有 301 个与该提交完全一致，其余 27 个是修改或新增的文件。
- `verl/version/version` 中的版本号是 `0.5.0.dev`，与该提交所处的时间段吻合。

复现方法：

```bash
git clone https://github.com/volcengine/verl.git upstream-verl
cd upstream-verl
git show a7b8675:verl/trainer/fsdp_sft_trainer.py > /tmp/base.py
diff -u /tmp/base.py <DR-Venus>/SFT/verl/trainer/fsdp_sft_trainer.py
```

---

## 3. 改动文件分类

| 类别 | 文件 | 和 SFT 的关系 |
|---|---|---|
| **SFT 训练器** | `verl/trainer/fsdp_sft_trainer.py` | ✅ 唯一真正影响 SFT 的改动，见第 4 节 |
| 运维和日志 | `utils/flops_counter.py`、`utils/profiler/performance.py` | 与训练结果无关，见第 5 节 |
| RL：奖励函数 | 新增 `reward_score/llm_judge.py`、`format_and_f1.py`、`stock_judge.py`、`fact_test.py`；修改 `reward_score/__init__.py`、`ppo/reward.py` | ❌ SFT 用不到 |
| RL：奖励管理器 | 新增 `reward_manager/naive_batch.py`、`group_async_batch.py`；修改 `naive.py`、`__init__.py` | ❌ |
| RL：算法和训练循环 | `ppo/core_algos.py`、`ppo/ray_trainer.py`、`utils/dataset/rl_dataset.py`、`workers/rollout/vllm_rollout/vllm_rollout_spmd.py`、`workers/config/rollout.py`，以及 ppo / rollout / reward_model / data 相关的 yaml | ❌ |

`SFT/verl` 中有这么多 RL 改动，是因为它是从 RL 的 verl 复制出来的一个中间版本。它与 `RL/verl` 也不完全相同，例如 RL 那边多了 `trainer/ppo/info_gain_advantage.py`（IGPO 算法）。这些对 SFT 都没有影响。

---

## 4. SFT 训练器的两处修改

文件：[SFT/verl/trainer/fsdp_sft_trainer.py](../SFT/verl/trainer/fsdp_sft_trainer.py)

### 4.1 关闭 sampler 的随机打乱

位置：[fsdp_sft_trainer.py:175-177](../SFT/verl/trainer/fsdp_sft_trainer.py#L175-L177)

```diff
 self.train_sampler = DistributedSampler(
-    self.train_dataset, shuffle=True,  num_replicas=world_size, rank=rank, drop_last=True
+    self.train_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
 )
```

**作用**：训练时严格按照 parquet 文件中的行顺序读取数据。

**影响**：打乱这件事完全交给了数据清洗阶段的 `prepare_trajectories.py --shuffle --seed 42`，两者必须配套使用。

- 清洗时如果**忘了加 `--shuffle`**，重采样复制出的 2 份或 5 份副本会**连续出现**，同一个 batch 里可能全是同一条轨迹；数据也会按源文件的顺序排列。
- 好处是数据顺序完全可复现，断点续训时 `StatefulDataLoader` 恢复的位置也是确定的。

### 4.2 `total_training_steps` 同时作用于学习率调度，并开放更多调度参数

位置：[fsdp_sft_trainer.py:334-367](../SFT/verl/trainer/fsdp_sft_trainer.py#L334-L367)

```diff
 self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
+if self.config.trainer.total_training_steps is not None:
+    self.total_steps = int(self.config.trainer.total_training_steps)

 num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
+min_lr_ratio = float(self.config.optim.get("min_lr_ratio", 0.0))
+num_cycles   = float(self.config.optim.get("num_cycles", 0.5))
+stable_ratio = float(self.config.optim.get("stable_ratio", 0.9))

 if ... lr_scheduler == "cosine":
     get_cosine_schedule_with_warmup(..., num_training_steps=self.total_steps,
+                                    min_lr_ratio=min_lr_ratio, num_cycles=num_cycles)
 elif lr_scheduler == "wsd":
     get_wsd_schedule_with_warmup(..., num_training_steps=self.total_steps,
+                                 min_lr_ratio=min_lr_ratio, num_cycles=num_cycles,
+                                 stable_ratio=stable_ratio)
```

**作用一：修复上游的不一致。**

上游代码中，`total_training_steps` 只决定训练在哪一步停止（[fsdp_sft_trainer.py:726-772](../SFT/verl/trainer/fsdp_sft_trainer.py#L726-L772)），学习率调度器却仍然按 `epoch 数 × 每个 epoch 的步数` 计算总步数。

例子：设置 `TOTAL_TRAINING_STEPS=100` 做快速试跑，而按 epoch 计算的总步数是 585。上游代码中训练在第 100 步停止，但调度器以为总共有 585 步：预热要到第 58 步才结束，到第 100 步时 cosine 衰减才刚开始。修改后两者使用同一个总步数（100），预热在第 10 步结束，第 100 步时学习率正好衰减到底。

**作用二：开放三个调度参数。**

底层函数（[SFT/verl/utils/torch_functional.py](../SFT/verl/utils/torch_functional.py) 中的 `get_cosine_schedule_with_warmup` 和 `get_wsd_schedule_with_warmup`）原本就支持这三个参数，但上游训练器没有传进去：

| 参数 | 默认 | 含义 |
|---|---|---|
| `min_lr_ratio` | 0.0 | 衰减到最后的最低学习率，占峰值学习率的比例。0 表示衰减到 0。 |
| `num_cycles` | 0.5 | cosine 的周期数。0.5 表示半个周期，即从峰值单调下降到最低点。 |
| `stable_ratio` | 0.9 | 只用于 WSD（Warmup-Stable-Decay）调度：稳定阶段（保持峰值学习率）占的比例。 |

> **注意**：这三个参数**没有**写进 [sft_trainer.yaml](../SFT/verl/trainer/config/sft_trainer.yaml)，代码中用 `.get(...)` 读取并给出默认值，所以默认运行时行为与上游完全相同。Hydra 配置默认是 struct 模式，不允许直接设置 yaml 中不存在的 key，因此需要加 `+` 前缀：
>
> ```bash
> bash train_sft.sh +optim.min_lr_ratio=0.1 optim.lr_scheduler=wsd +optim.stable_ratio=0.8
> ```

`sft_shells/run.sh` 没有设置这些参数，所以按论文配置训练时，使用的是默认的 cosine 调度：前 10% 步数线性预热，之后按半个余弦周期衰减到 0。

---

## 5. 其他两处小改动

| 文件 | 改动 | 作用 |
|---|---|---|
| [flops_counter.py:101](../SFT/verl/utils/flops_counter.py#L101) | 新增 `"PPU-ZW810E"` 的峰值算力 123 TFLOPS | 让 MFU（算力利用率）指标在这种加速卡上也能正确计算。说明作者的训练硬件很可能是这种 PPU 卡，而不是 NVIDIA GPU。 |
| [profiler/performance.py:147](../SFT/verl/utils/profiler/performance.py#L147) | `marked_timer` 在 rank 0 上打印每个阶段的开始和耗时 | 只影响调试日志，不改变训练逻辑。 |

---

## 6. 直接沿用上游的部分

SFT 真正依赖的核心功能都来自上游，没有改动：

| 功能 | 上游实现 | 本项目怎么使用 |
|---|---|---|
| 多轮对话数据 + 只对 assistant 计算损失 | [multiturn_sft_dataset.py](../SFT/verl/utils/dataset/multiturn_sft_dataset.py) | 数据清洗把轨迹整理成 `messages` 列；工具定义直接写进 system prompt，所以不需要 `tools` 列 |
| 长上下文 | Ulysses 序列并行 + remove padding + gradient checkpointing | `train_sft.sh` 传入 `SP_SIZE=8`、`max_length=200000` |
| FSDP2 分片训练与断点续训 | `fsdp_sft_trainer.py` 与 `FSDPCheckpointManager` | `SAVE_FREQ=100`，`RESUME_MODE=auto` |
| 合并 checkpoint | [verl/model_merger](../SFT/verl/model_merger/) | 通过 `scripts/merge_checkpoint.sh` 调用，见 [sft-script.md §2](sft-script.md) |
