# 12 周学习与实战计划（2026-08-17—2026-11-08）

目标投入为每周 18 小时、总计约 216 小时；预算目标 CNY 2,600，硬上限
CNY 3,000。本计划的完成标准不是复述论文或跑通一条命令，而是能推导、
改动、实验、定位失败，并用个人生成的证据接受面试追问。

## 掌握等级与固定节奏

- L4：能推导、改代码、设计对照、解释失败和边界。
- L3：能画数据流、配置运行并排查常见显存、并发和分布式问题。
- L2：知道模块边界与调用入口，不要求重写 vendored 框架。

最终要求：轨迹治理、多轮 SFT、Agent Loop、GRPO/IGPO、奖励与评测达到
L4；FSDP、Ulysses SP、Ray、vLLM、KV cache、异步 rollout 达到 L3；其余
vendored `verl` 通用代码达到 L2，不顺序通读全部源码。

每周固定安排：周一论文与概念 2h；周二公式推导 2h；周三核心调用链 2h；
周四测试、笔记或实现 2h；周六实验 6h；周日分析、复盘和提交周产物 4h。

## 逐周验收

| 周次 | 本周学习与工作 | 掌握/验收标准 | 必须提交的证据 |
|---|---|---|---|
| 1（8/17–8/23） | 通读 DR-Venus；理解 Deep Research、搜索/访问工具与 SFT→RL→推理闭环；核对上游与个人贡献 | 无稿讲清完整链路 10 分钟；解释小模型为何更怕噪声、长程任务为何需要密集信用分配；能指出 DR-Venus、IGPO、verl 的边界 | 架构图、术语表、贡献边界、仓库审计；见 `architecture.md`、`glossary.md`、`contribution-boundaries.md`、`repository-audit.md` |
| 2（8/24–8/30） | 用官方 SFT/RL 权重做推理；阅读 Agent 状态机、工具协议、解析、超时、去重、强制结束；建立 online/record/replay | 从 question 追踪到 `<answer>`；人为制造格式错、工具超时、重复调用并定位恢复路径 | 20 题 smoke 原始轨迹；完成率/格式率/轮数/延迟/API 成本；3–5 分钟 Demo |
| 3（8/31–9/6） | 吃透归一化、非法工具剪枝、结构校验和 1×/2×/5×重采样；执行 correctness filtering；先 split 后重采样 | 解释每个 drop reason；说明重采样不创造新信息及其过拟合风险；拒绝 `VAL_FILES=TRAIN_FILES` | 全量数据审计；question-level 90/5/5；人工抽检 100 条且 Judge/人工一致率目标 ≥90%；确定性 `data_manifest.json` |
| 4（9/7–9/13） | 多轮 chat template、teacher forcing、assistant-only loss mask、截断、LoRA、checkpoint、FSDP/Ulysses；Qwen3-0.6B smoke | 手工标出 CE token；解释工具响应只作为条件；能定位 mask 错位、空监督与截断问题 | 100 条、2K–4K context、50-step smoke；mask/loss 可视化；resume 验证 |
| 5（9/14–9/20） | 4B SFT：S0 均匀采样与 S1 长轨迹加权采样 | 解释 LoRA/全参/FSDP/SP 显存取舍；从 OOM 判断参数、优化器、激活或 KV cache 瓶颈 | 两个 LoRA adapter；resolved command/config/data/model hash；训练曲线、GPU-hour、成本和失败日志 |
| 6（9/21–9/27） | 评测个人 SFT；深入 DP/FSDP、TP、SP、vLLM 与 KV cache | 画出 2 卡 SFT、4 卡 RL 数据流；说清参数/梯度/优化器/激活/KV 的位置 | Base/S0/S1 对照；至少 30 个失败样本分类；SFT 阶段报告 |
| 7（9/28–10/4） | 从 PPO clipped objective 推到 GRPO，再到 IGPO；学习 advantage collapse、GT log-prob 差、归一化、折扣、KL、格式惩罚 | 三轮例子手算 `IG_t`、group normalization 与 token advantage；回答 shaping、`gamma=0.95`、Judge hacking | 公式笔记；5 组 toy batch；reference 与生产代码误差 ≤1e-6 |
| 8（10/5–10/11） | 跟踪 `main_ppo → Ray rollout → tool → reward → IG → advantage → actor update`；3-step RL smoke | 解释 response mask 与 policy-loss mask；解释 tool-response token 为何承载 advantage 却不进 actor loss；排查全零 reward、边界错位、NaN | 3-step 日志；完成率目标 ≥90%、格式率 ≥95%、visit-turn IG 对齐率 ≥95%、NaN/Inf=0 |
| 9（10/12–10/18） | 4B R0/R1/R2 × seed 42/43 短程消融 | 从 outcome diversity、IG、KL、格式错误、browse ratio 判断 collapse、turn shortening 或 reward hacking | 六组日志/checkpoint/manifest；同预算表；失败组同样保留并解释 |
| 10（10/19–10/25） | 冻结工具与 Judge cache；比较官方 SFT/RL 与个人模型；Pass@1、固定 20 题 Pass@4、bootstrap CI、人工校准、sync/async profiling | 区分训练收益、采样方差、Judge 噪声与网页漂移；不凭单次偶然结果下结论 | 结果表和 95% CI；原始 predictions；50 个失败案例；吞吐/显存/API/GPU 成本表 |
| 11（10/26–11/1） | 整理公开仓库、依赖锁、`.env.example`、一键 smoke、数据/模型/实验 manifest；报告与 Demo | 全新环境 30 分钟内完成 CPU/smoke；每个简历数字能追到 config、日志和原始预测 | 8–12 页报告、仓库、3–5 分钟 Demo、架构/结果图、简历初稿 |
| 12（11/2–11/8） | 准备 60 秒、5 分钟、20 分钟讲述；三次模拟面试；红队个人贡献、泄漏、成本、失败实验 | 白板推导 assistant-only SFT 与 IGPO；现场定位代码；不回避负结果；明确所有归因 | 25 道追问题库、三次模拟记录、最终 bullets、讲稿与答辩材料 |

## 实验与简历硬门槛

正式 SFT 固定同一 train/validation split、同一 20M assistant-token target，完整
batch 后 S0/S1 实际监督 token 差不超过 target 的 0.1%。正式 RL 固定 96 train、
32 validation、4×A100-80G、32K context、12 turns、group/train batch=4、20
steps，运行 R0/GRPO、R1/IGPO、R2/IGPO+格式惩罚与两个 seed。所有付费任务先
跑 3–5 step pilot，预算闸门不允许 CNY 3,000 以上的投影。

只有同时满足下列条件，才能写进简历：亲自跑通官方 checkpoint Agent 闭环；
完成真实 4B SFT 与至少一组 GRPO/IGPO 对照；保留评测脚本、原始预测、日志、
成本和失败分析；能推导 IGPO；公开归因 DR-Venus、IGPO、verl。官方 29.1 等
结果只作背景，个人结果必须写明 N、max-turn、context、seed 与置信区间。若没有
稳定涨分，就量化并解释失败机制，不替换成官方数字。

每周状态以 `progress.md` 为准。代码存在只表示工具已就绪；GPU/API 运行、人工
审阅和口头掌握必须由学习者真实完成后才能勾选。
