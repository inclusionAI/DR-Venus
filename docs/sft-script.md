# SFT 训练脚本与参数详解

> 本文说明 `SFT/` 目录下各个脚本的作用、checkpoint 合并的原理，以及核心训练脚本 `train_sft.sh` 的全部可选参数。
> 数据清洗部分见 [sft-data.md](sft-data.md)，整体流程见 [overview.md](overview.md)。

---

## 1. 目录与脚本的作用

调用关系：

```
run.sh ──► sft_shells/run.sh ──► train_sft.sh ──► torchrun verl.trainer.fsdp_sft_trainer (verl/)
                                                         │ 每隔 SAVE_FREQ 步保存
                                                         ▼
                                    分片 checkpoint ──► scripts/merge_checkpoint.sh ──► HF 格式模型
data_clean/ ──► cleaned_resampled.parquet ──┘（作为 TRAIN_FILES 输入）
```

| 路径 | 作用 |
|---|---|
| [verl/](../SFT/verl/) | 训练框架本身，是字节开源 verl 的一个副本。SFT 实际使用 [verl/trainer/fsdp_sft_trainer.py](../SFT/verl/trainer/fsdp_sft_trainer.py)：用 FSDP 把模型切到多张 GPU 上，用 Ulysses 序列并行支持 200K 长度的上下文，多轮数据只对 assistant token 计算损失。 |
| [train_sft.sh](../SFT/train_sft.sh) | **SFT 的核心入口**。从环境变量读取配置，拼成 `torchrun ... -m verl.trainer.fsdp_sft_trainer data.xxx=... model.xxx=...` 这样的 Hydra 命令并执行，同时把日志写到 `${CKPT_DIR}`。设置 `DRY_RUN=True` 时只打印命令，不实际执行。 |
| [sft_shells/run.sh](../SFT/sft_shells/run.sh) | **带默认配置的封装**，填好了论文的长上下文设置：基座 `Qwen3-4B-Thinking-2507`、`DATA_MAX_LENGTH=200000`、`SP_SIZE=8`、全局 batch 32、每卡 micro batch 1、1 个 epoch、每 100 步保存和验证一次。这些值都写成 `${VAR:-默认值}`，可以用环境变量覆盖。最后调用 `train_sft.sh`。 |
| [run.sh](../SFT/run.sh) | 根目录下的快捷入口，只有一行 `exec sft_shells/run.sh "$@"`，方便在 `SFT/` 下直接运行 `bash run.sh`。 |
| [data_clean/](../SFT/data_clean/) | 把小红书（RED）的 REDSearcher 轨迹清洗成 SFT 能用的数据，产出 `cleaned_resampled.parquet`。详见 [sft-data.md](sft-data.md)。 |
| [scripts/merge_checkpoint.sh](../SFT/scripts/merge_checkpoint.sh) | 把训练时保存的**分片 checkpoint 合并成 Hugging Face 格式**，见第 2 节。 |

---

## 2. Checkpoint 合并

### 2.1 为什么需要合并

训练使用 **FSDP**（全分片数据并行）：模型参数被切成若干片，分别放在不同的 GPU 上。保存 checkpoint 时，每张卡只保存自己那一片，所以每个 `global_step_N/` 目录里不是一个完整模型，而是下面这样一组文件：

```
global_step_100/
├── model_world_size_8_rank_0.pt        ← 第 0 张卡上的参数分片
├── ...
├── model_world_size_8_rank_7.pt
├── optim_world_size_8_rank_*.pt        ← 优化器状态分片（只在断点续训时用）
├── extra_state_world_size_8_rank_*.pt  ← 学习率调度器和随机数状态
├── data.pt                             ← dataloader 进度
└── huggingface/                        ← 只有 config 和 tokenizer，没有权重
```

这种格式只有 verl 在相同并行配置下才能读取，用来**断点续训**。vLLM、`transformers.from_pretrained` 和下一阶段的 RL 训练都无法直接加载。而 SFT 的默认配置 `save_contents: ["model", "optimizer", "extra"]`（[sft_trainer.yaml:81](../SFT/verl/trainer/config/sft_trainer.yaml#L81)）不会额外保存完整的 HF 权重，所以必须做一次合并。

### 2.2 合并过程

`merge_checkpoint.sh` 只负责解析参数，然后调用：

```bash
python -m verl.model_merger merge --backend fsdp --local_dir <分片目录> --target_dir <输出目录>
```

核心逻辑在 `FSDPModelMerger.merge_and_save`（[fsdp_model_merger.py:206](../SFT/verl/model_merger/fsdp_model_merger.py#L206)）：

1. **确定分片数量**：从文件名 `model_world_size_{N}_rank_*.pt` 中读出 N。
2. **读取 rank 0 的分片**，根据其中的 DTensor 推断设备网格（mesh），即参数沿哪个维度、被切成了几份。
3. **读取所有 rank 的分片**，对每个参数按切分方式（placement）把各片**拼接还原**（`_merge_by_placement`），得到完整的 state dict。
4. **保存为 HF 格式**：结合 `huggingface/` 中的 config 和 tokenizer，写出 `config.json`、`*.safetensors` 和 tokenizer 文件。加 `--hf_upload` 时还会上传到 Hub。

如果把 `merge` 换成 `test`，它会把合并结果和一个参考 HF 模型逐个参数对比，用来验证合并是否正确。

### 2.3 合并之后的用途

合并后得到的是标准 HF 模型目录，也就是论文中的 **DR-Venus-4B-SFT**：

- 作为 RL 阶段的初始模型，填到 `RL/train_igpo.sh` 的模型路径中。
- 交给 vLLM 部署，在 `Inference/` 下做评测。
- 上传到 Hugging Face 发布。

### 2.4 使用时注意

脚本 help 中的示例路径是 `.../global_step_100/actor`，这是 **RL** checkpoint 的目录结构（actor 是 RL 中的策略模型）。从 SFT 的 `save_checkpoint`（[fsdp_sft_trainer.py:541-557](../SFT/verl/trainer/fsdp_sft_trainer.py#L541-L557)）看，SFT 的分片直接保存在 `${CKPT_DIR}/global_step_N/` 下，没有 `actor/` 子目录。合并 SFT 模型时应该这样写：

```bash
bash scripts/merge_checkpoint.sh --backend fsdp \
  --local-dir ./outputs/sft/qwen3-4b-sft/global_step_<N> \
  --target-dir ./merged/DR-Venus-4B-SFT
```

> 这条路径是根据代码推断的，没有用真实 checkpoint 验证过。使用前先 `ls` 该目录，确认里面有 `model_world_size_*_rank_*.pt`。`--backend megatron` 用于 Megatron 训练的 checkpoint，本项目的 SFT 用不到。

也可以在训练时直接保存 HF 权重，从而跳过合并（见第 3.8 节的 `hf_model`）。

---

## 3. `train_sft.sh` 参数详解

`train_sft.sh` 读取环境变量，翻译成 verl 的 Hydra 参数（`key=value` 形式），再用 `torchrun` 启动 `verl.trainer.fsdp_sft_trainer`。下面按用途分组列出全部参数：“Hydra 参数”是实际传给 verl 的名字，“论文设置”是 [sft_shells/run.sh](../SFT/sft_shells/run.sh) 中填好的值。

### 3.1 分布式启动

| 变量 | 默认 | Hydra 参数 | 作用 |
|---|---|---|---|
| `ENTRYPOINT` | `-m verl.trainer.fsdp_sft_trainer` | — | torchrun 运行的 Python 入口，一般不需要改。 |
| `NNODES` | 1 | torchrun `--nnodes` | 使用几台机器。 |
| `NODE_RANK` | 0 | `--node_rank` | 当前机器的编号。多机训练时每台机器设不同的值，主节点为 0。 |
| `NUM_GPUS` | 8 | `--nproc_per_node` | 每台机器使用几张 GPU。 |
| `MASTER_ADDR` / `MASTER_PORT` | 127.0.0.1 / 29500 | `--master_addr/port` | 主节点的地址和端口。多机训练时改成主节点 IP。 |

### 3.2 模型与数据（必填项在这里）

| 变量 | 默认 | Hydra 参数 | 作用 | 论文设置 |
|---|---|---|---|---|
| `MODEL_PATH` | **必填** | `model.partial_pretrain` | 基座模型的本地路径或 HF ID。 | `Qwen/Qwen3-4B-Thinking-2507` |
| `TRAIN_FILES` | **必填** | `data.train_files` | 训练 parquet，即 data_clean 产出的 `cleaned_resampled.parquet`。 | `${DATA_ROOT}/train.parquet` |
| `VAL_FILES` | 等于 `TRAIN_FILES` | `data.val_files` | 验证集。不设置时直接用训练集做验证。 | `${DATA_ROOT}/val.parquet` |
| `DATA_MAX_LENGTH` | 32768 | `data.max_length` | 单条样本的最大 token 数。 | **200000** |
| `TRUNCATION` | `right` | `data.truncation` | 超过 `max_length` 时怎么处理：`right` 保留开头、截掉末尾；`left` 保留末尾；`error` 直接报错。 | `right` |

> **关于 `TRUNCATION=right`**：对 Deep Research 轨迹来说，被截掉的末尾恰好是**最终的 `<answer>`**，模型学不到这条轨迹的结论（[multiturn_sft_dataset.py:338-347](../SFT/verl/utils/dataset/multiturn_sft_dataset.py#L338-L347)）。想知道有多少样本超长，可以先把 `TRUNCATION` 设成 `error` 跑一遍，或者在数据清洗阶段统计 token 长度。

### 3.3 多轮数据格式

| 变量 | 默认 | Hydra 参数 | 作用 |
|---|---|---|---|
| `MULTITURN` | `True` | `data.multiturn.enable` | 打开后使用 `MultiTurnSFTDataset`：读取 `messages` 列，用 chat template 渲染，**只对 assistant 的 token 设置 loss_mask=1**。本项目必须打开。 |
| `MESSAGES_KEY` | `messages` | `data.multiturn.messages_key` | 对话列表在 parquet 中的列名，与 data_clean 的输出一致。 |
| `TOOLS_KEY` | `tools` | `data.multiturn.tools_key` | 工具定义所在的列名，有这一列时会传给 chat template 自动生成工具说明。本项目数据**没有**这一列（工具定义已直接写在 system prompt 中），所以不起作用。 |
| `ENABLE_THINKING_KEY` | `enable_thinking` | `data.multiturn.enable_thinking_key` | 每行是否开启思考模式的列名，会传给 Qwen3 chat template 的 `enable_thinking`。本项目数据中也没有这一列。 |

以下四个参数**只在单轮模式（`MULTITURN=False`）下生效**，本项目用不到：

| 变量 | 默认 | 作用 |
|---|---|---|
| `PROMPT_KEY` | `question` | 单轮模式下 prompt 所在的列。 |
| `RESPONSE_KEY` | `gt` | 单轮模式下 response 所在的列。 |
| `PROMPT_DICT_KEYS` / `RESPONSE_DICT_KEYS` | null | 该列存的是字典时，用这里指定的 key 取出文本。 |

### 3.4 Batch 与优化

| 变量 | 默认 | Hydra 参数 | 作用 | 论文设置 |
|---|---|---|---|---|
| `TRAIN_BATCH_SIZE` | 32 | `data.train_batch_size` | **全局** batch size，即每次参数更新使用的样本数。 | 32 |
| `MICRO_BATCH_SIZE_PER_GPU` | 1 | `data.micro_batch_size_per_gpu` | 每张卡每次前向/反向处理的样本数，也用作验证 batch size。 | 1 |
| `LR` | 1e-5 | `optim.lr` | 峰值学习率。 | 1e-5 |
| `TOTAL_EPOCHS` | 1 | `trainer.total_epochs` | 训练几个 epoch。 | 1 |
| `TOTAL_TRAINING_STEPS` | null | `trainer.total_training_steps` | 总训练步数，设置后覆盖按 epoch 计算的步数，适合快速试跑。 | — |

**参数之间的换算**（[fsdp_sft_trainer.py:139-149](../SFT/verl/trainer/fsdp_sft_trainer.py#L139-L149)）：

```
数据并行组数 dp = 总 GPU 数 / SP_SIZE = 8 / 8 = 1
每个 dp 组的 batch = TRAIN_BATCH_SIZE / dp = 32
梯度累积次数 = 32 / MICRO_BATCH_SIZE_PER_GPU = 32
每个 epoch 的步数 ≈ 18,745 / 32 ≈ 585
```

论文配置下，8 张卡**合起来处理一条序列**，每次参数更新累积 32 条样本的梯度。代码要求 `TRAIN_BATCH_SIZE` 能被 dp 整除，dp 组内的 batch 也要能被 micro batch 整除，否则报错。

脚本没有暴露的优化器参数使用 [sft_trainer.yaml](../SFT/verl/trainer/config/sft_trainer.yaml) 中的默认值：AdamW，betas (0.9, 0.95)，weight_decay 0.01，cosine 学习率调度，前 10% 步数预热，梯度裁剪 1.0。

### 3.5 显存与长上下文

| 变量 | 默认 | Hydra 参数 | 作用 | 论文设置 |
|---|---|---|---|---|
| `SP_SIZE` | 1 | `ulysses_sequence_parallel_size` | **Ulysses 序列并行**：把一条序列切成 `SP_SIZE` 段，分到不同卡上计算注意力。200K 长度单卡放不下，必须开启。 | **8** |
| `RM_PAD` | `True` | `use_remove_padding` | 去掉 padding 再计算，把变长序列拼接起来，不在 pad token 上浪费算力。**`SP_SIZE>1` 时必须打开**，代码中有断言（[fsdp_sft_trainer.py:226](../SFT/verl/trainer/fsdp_sft_trainer.py#L226)）。 | True |
| `LIGER` | `False` | `model.use_liger` | 使用 Liger Kernel 的融合算子（RMSNorm、交叉熵等），省显存、提速度，需要另外安装 `liger-kernel`。 | False |
| `LORA_RANK` | 0 | `model.lora_rank` | 大于 0 时改用 LoRA 微调，0 表示全参数微调。 | 0（全参数） |
| `LORA_ALPHA` | 16 | `model.lora_alpha` | LoRA 缩放系数，只在 LoRA 模式下生效。 | — |
| `TARGET_MODULES` | `all-linear` | `model.target_modules` | LoRA 挂在哪些层上，只在 LoRA 模式下生效。 | — |

另外，脚本固定打开了 `model.enable_gradient_checkpointing=true`，用重新计算换取显存，这对长上下文训练是必需的。

### 3.6 输出、日志与保存

| 变量 | 默认 | Hydra 参数 | 作用 | 论文设置 |
|---|---|---|---|---|
| `PROJECT_NAME` | `deepresearcher-sft` | `trainer.project_name` | 实验分组名，用于 tensorboard / wandb。 | 同默认 |
| `EXP_NAME` | `模型名-时间戳` | `trainer.experiment_name` | 本次实验的名字。 | `qwen3-4b-sft` |
| `OUTPUT_ROOT` | `./outputs/sft` | — | 输出根目录。 | — |
| `CKPT_DIR` | `${OUTPUT_ROOT}/${EXP_NAME}` | `trainer.default_local_dir` | checkpoint 和日志的保存目录，`global_step_N/` 就在这里。 | — |
| `LOG_FILE` | `${CKPT_DIR}/train-时间戳.log` | — | 终端输出通过 `tee` 同时写入这个文件。注意：help 文本中没有列出这个变量。 | — |
| `LOGGER` | `['console','tensorboard']` | `trainer.logger` | 日志输出位置，可以加上 `'wandb'`。 | — |
| `SAVE_FREQ` | -1 | `trainer.save_freq` | 每多少步保存一次 checkpoint，-1 表示中途不保存，训练结束时保存。 | 100 |
| `TEST_FREQ` | -1 | `trainer.test_freq` | 每多少步做一次验证，-1 表示不做。 | 100 |

### 3.7 断点续训

| 变量 | 默认 | 作用 |
|---|---|---|
| `RESUME_MODE` | `auto` | `auto`：在 `CKPT_DIR` 中找最新的 `global_step_N`，有就接着训练；`disable`：从头开始；`resume_path`：从 `RESUME_FROM_PATH` 恢复。 |
| `RESUME_FROM_PATH` | null | 要恢复的 checkpoint 路径，必须包含 `global_step_`。 |

> **注意**：`EXP_NAME` 的默认值带时间戳，直接运行 `train_sft.sh` 时每次的 `CKPT_DIR` 都是新目录，`auto` 模式**永远找不到之前的 checkpoint**。要自动续训，就固定 `EXP_NAME` 或 `CKPT_DIR`。`sft_shells/run.sh` 已把 `EXP_NAME` 固定为 `qwen3-4b-sft`，用它启动时没有这个问题。

### 3.8 调试与扩展

| 变量 / 方式 | 作用 |
|---|---|
| `DRY_RUN=True` | 只打印最终的 torchrun 命令，不启动训练，用于检查参数。 |
| `EXTRA_TRAIN_ARGS` | 额外的 Hydra 参数，按空格切分后追加到命令末尾。 |
| 命令行位置参数 `"$@"` | 同样追加到命令末尾，且排在**最后，优先级最高**。 |

后两种方式可以设置脚本没有暴露的配置，例如：

```bash
bash train_sft.sh trainer.max_ckpt_to_keep=2 optim.warmup_steps_ratio=0.05 \
  'trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model]'
```

最后一项加上 `hf_model` 后，保存 checkpoint 时会额外保存一份完整的 HF 格式模型，不需要再运行 `merge_checkpoint.sh`，代价是占用更多磁盘空间。

脚本中固定写死、不能通过环境变量修改的两项：`model.enable_gradient_checkpointing=true`、`trainer.default_hdfs_dir=null`。如需修改，同样通过位置参数覆盖。

---

## 4. 复现论文的最小命令

```bash
cd SFT
DATA_ROOT=/path/to/data \
TRAIN_FILES=/path/to/cleaned_resampled.parquet \
VAL_FILES=/path/to/val.parquet \
DRY_RUN=True bash run.sh     # 先确认命令无误，再去掉 DRY_RUN
```

`sft_shells/run.sh` 已填好论文的长上下文配置，一般只需提供数据路径，其他参数按需用环境变量覆盖。训练结束后，按第 2.4 节合并最后一个 `global_step_N`，得到 DR-Venus-4B-SFT。
