# verl 版本演进与 DR-Venus 的对应关系

> 本文说明 DR-Venus 使用的 verl 处于哪个版本，上游之后各版本新增了哪些主要功能，以及升级对本项目的影响。
> 项目对 verl 做了哪些修改，见 [sft-verl.md](sft-verl.md)。
> 版本信息统计截至 2026-10-21。

---

## 1. 当前版本与最新版本

| | 版本 | 时间 |
|---|---|---|
| **DR-Venus 使用的 verl** | `0.5.0.dev`，上游 main 分支提交 `a7b8675` | 2025-09-15 |
| 前一个正式版 | v0.5.0 | 2025-07-23 |
| 之后的第一个正式版 | v0.6.0（第一个包含该提交的 tag） | 2025-10-15 |
| **最新正式版** | **v0.9.1** | 2026-09-20 |
| main 最新提交 | `cd58f550` | 2026-10-20 |

DR-Venus 使用的不是某个正式版本，而是 **v0.5.0 和 v0.6.0 之间的一个开发快照**，最接近 v0.6.0 发布前一个月的代码。到 2026-10，它已经落后 **4 个大版本、约一年**。

> 仓库已从 `volcengine/verl` 迁移到 **[verl-project/verl](https://github.com/verl-project/verl)**，旧地址会自动跳转。

### 如何确认

```bash
cat SFT/verl/version/version            # 0.5.0.dev

git clone https://github.com/verl-project/verl.git upstream-verl
cd upstream-verl
git log -1 --format='%cs %s' a7b8675    # 基线提交的日期和标题
git tag --contains a7b8675 | sort -V    # 包含该提交的正式版：v0.6.0 起
git for-each-ref --sort=creatordate \
  --format='%(refname:short) %(creatordate:short)' refs/tags   # 所有版本及发布时间
```

基线提交的确定方法见 [sft-verl.md §2](sft-verl.md)。

---

## 2. 版本时间线

| 版本 | 发布时间 | 一句话概括 |
|---|---|---|
| v0.4.0 / v0.4.1 | 2025-06 | — |
| v0.5.0 | 2025-07-23 | — |
| ▶ **DR-Venus 基线** | 2025-09-15 | `0.5.0.dev` 开发快照 |
| v0.6.0 | 2025-10-15 | Model Engine 原型，Rollout 改为 Server 模式 |
| v0.6.1 | 2025-11-14 | 修复版本 |
| v0.7.0 | 2026-01-05 | 新引擎可用于生产，移除 SPMD rollout |
| v0.7.1 | 2026-03-16 | 删除 `fsdp_sft_trainer.py` |
| v0.8.0 | 2026-06-01 | On-Policy Distillation，废弃旧 worker |
| v0.9.0 | 2026-08-14 | 统一的 V1 trainer 默认启用，硬件插件化 |
| v0.9.1 | 2026-09-20 | V1 trainer 增强，Liger 融合交叉熵 |

---

## 3. 各版本主要功能

以下根据各版本 [release notes](https://github.com/verl-project/verl/releases) 的 Highlights 部分整理，只列重点。

### v0.6.0（2025-10）

- **Model Engine 原型**：把分布式模型封装成服务，提供统一 API，后端可以是 FSDP + Ulysses 或 Megatron。**最先基于它实现的就是 SFT trainer**。
- **Rollout 改为 Server 模式**：vLLM 和 SGLang 都改成原生的推理服务模式，更适合多轮对话和工具调用的 agentic RL；支持大型 MoE 模型的 DP+EP。
- 新增模型：Qwen3-VL、GPT-OSS。
- 新增算法：GSPO；token 级和序列级的 TIS（修正 rollout 与训练之间的分布差异）。
- 新增示例项目（recipe）：retool、deepeyes 等。

### v0.7.0（2026-01）/ v0.7.1（2026-03）

- **新的 Model Engine 可用于生产**：FSDP 和 Megatron 引擎都已稳定；支持 VLM 的 SFT 和 RL；新增 VeOmni 引擎（alpha 阶段）。
- 接入 Megatron-Bridge，支持 LoRA/PEFT 和实验性的 FP8 训练。
- **移除 SPMD 模式的 rollout**，只保留 Server 模式。
- Reward 服务化：支持生成式、判别式、规则型奖励以及它们的组合。
- 新增算法：CISPO、SAPO。
- v0.7.1：SFT 和 RL 支持 MTP 训练；**删除 `fsdp_sft_trainer.py`**（PR #5382），SFT 统一改用基于引擎的 `sft_trainer.py` / `sft_trainer_ray.py`。

### v0.8.0（2026-06）

- **On-Policy Distillation（OPD）**：在 FSDP、Megatron、VeOmni 三种后端上端到端支持，可以用多个 teacher，支持同步和全异步。
- 新的 sync trainer，用 TransferQueue 把控制流和数据流分开，多模态训练最多快 2 倍。
- Megatron：Megatron-FSDP 模式、动态 context parallel、NVFP4 QAT。
- Rollout：SGLang 支持 LoRA 和 Prefill-Decode 分离；TensorRT-LLM 支持异步 RL。
- 新的 agent 项目 **[uni-agent](https://github.com/verl-project/uni-agent)**，支持 coding、search、GUI agent 的 RL 训练。
- SFT 相关：DP 负载均衡；FSDP 引擎会校验 SP 与 remove_padding 的配置组合。
- **不兼容的改动**：旧的 FSDP/Megatron worker 被废弃，统一迁移到引擎抽象；`verl/interactions` 被废弃；`main_ppo.py` 被标记为废弃。

### v0.9.0（2026-08）/ v0.9.1（2026-09）

- **统一的 V1 trainer，默认启用**：`sync`、`colocate_async`、`separate_async` 三种模式共用一套控制流和 replay buffer；可以控制 off-policy 数据的陈旧程度。
- 大模型支持：DeepSeek-V4、GLM-5、Kimi-K2.5/K2.6；新增 Megatron Lite 后端；支持 Muon 优化器。
- vLLM 支持 Prefill-Decode 分离；rollout 可做到完全确定（两次相同的运行得到逐位一致的奖励曲线）。
- 新增 `delta_sharded` checkpoint 引擎：只传输变化的权重，权重同步快 2–3 倍。
- 硬件插件化：抽象出平台层，通过插件支持寒武纪、沐曦、天数、Intel XPU、AMD ROCm、昇腾等硬件。
- Agentic RL：Continuous Token 机制，保证多轮 rollout 中模型输出与工具返回之间的 token 衔接一致（默认关闭）。
- v0.9.1：Liger Kernel 融合交叉熵，actor 更新快 13.5%；新增 checkpoint 回调；修复 model merger 合并复制 buffer 时的问题；旧的 `fully_async_policy` 和 `one_step_off_policy` 被 V1 trainer 取代，已标记废弃。

---

## 4. 对 DR-Venus 的影响

### 4.1 SFT 部分

两个重要的 SFT 改动，DR-Venus 的基线**已经包含**：

| PR | 合入时间 | 内容 | 是否在基线中 |
|---|---|---|---|
| #3287 | 2025-09-03 | 修复 SFT loss mask 切片差一位的问题 | ✅ |
| #3341 | 2025-09-04 | 基于 model engine 的 SFT trainer | ✅（但项目仍使用旧的 `fsdp_sft_trainer.py`） |
| #5382 | 2026-02-24 | 删除 `fsdp_sft_trainer.py` | ❌ |

升级的影响：

- 升级到 **v0.7.1 及以后**，`fsdp_sft_trainer.py` 已不存在。项目对它的两处修改（关闭 sampler 打乱、学习率调度总步数，见 [sft-verl.md §4](sft-verl.md)）需要迁移到新的 `sft_trainer.py`，`train_sft.sh` 中的入口和部分 Hydra 参数也要跟着改。
- 新版本中对 SFT 有实际价值的功能：
  - DP 负载均衡：长短轨迹混在一起时，各张卡的计算量更均衡。
  - Liger 融合交叉熵：更快、更省显存。
  - 更完善的 checkpoint 管理和回调。

### 4.2 RL 部分

影响更大。DR-Venus 的 IGPO、`DRAgentLoop` 和奖励管理器都写在 2025-09 版本的接口上，而之后有多次不兼容的改动：

| 版本 | 不兼容的改动 |
|---|---|
| v0.7 | 移除 SPMD rollout |
| v0.8 | 废弃旧的 FSDP/Megatron worker、`verl/interactions`、`main_ppo.py` |
| v0.9 | 默认启用 V1 trainer，旧的异步训练方案被废弃 |

升级 RL 基本等于重新移植。

### 4.3 建议

- **只是复现论文**：继续使用项目自带的 verl，不要升级。
- **在本项目基础上长期开发**：SFT 可以先迁移，改动范围小；RL 部分等到需要新功能时再评估，例如 uni-agent、异步训练或 Continuous Token。
