# SFT 数据清洗流程详解

> 本文逐步解释 SFT 轨迹清洗流水线中每一步的作用，给出示例，并对照代码说明实现方式。整体位置见 [overview.md §1.1](overview.md)。
> 代码位置均指向 [SFT/data_clean/prepare_trajectories.py](../SFT/data_clean/prepare_trajectories.py)。

```
REDSearcher 原始轨迹 (10,001)
   │  SFT/data_clean/prepare_trajectories.py
   │    ① 格式对齐  ② 非法工具裁剪  ③ 重复调用去重  ④ 结构校验
   │  (+ 外部 LLM-judge 正确性过滤 → 9,365)
   │    ⑤ 按轮数重采样 (1x/2x/5x → 18,745)
   ▼
cleaned_resampled.parquet + stats.json
```

运行方式：

```bash
python SFT/data_clean/prepare_trajectories.py \
  --input <raw_parquet_or_dir> --output-dir <out> \
  --system-prompt-mode template --current-date 2026-03-01 --shuffle --seed 42
```

**执行顺序**：①–④ 在 `main()` 中按单条轨迹依次执行（[:807-855](../SFT/data_clean/prepare_trajectories.py#L807-L855)），全部轨迹通过后再统一执行 ⑤（[:880-910](../SFT/data_clean/prepare_trajectories.py#L880-L910)）。**LLM-judge 正确性过滤不在本脚本中**，需要另外实现。

---

## ① 格式对齐（Normalize）

### 作用

原始轨迹来自不同的生成过程，格式不统一：JSON 写法不规范，参数名混用（`query`/`queries`、`url`/`urls`），搜索结果是 markdown 文本，think 里混有 `<answer>`/`<summary>` 标签。这一步把它们统一成训练和推理都使用的格式：

```
system → user(问题) → [assistant(<think>+<tool_call>) → user(<tool_response>)]* → assistant(<think>+<answer>)
```

### 基本概念

一条训练样本就是一段对话，存成 `messages` 列表。列表中每个元素是一条消息，有两个字段：

| 字段 | 含义 |
|---|---|
| `role` | 这条消息是谁说的，取值有三种：`system`、`user`、`assistant` |
| `content` | 消息的文本内容 |

三种 `role` 在本项目中的含义：

| role | 在 Deep Research 场景中的含义 |
|---|---|
| `system` | **system prompt**（系统提示词），放在对话最前面。它告诉模型“你是深度研究助手，可以使用 `search`/`visit` 两个工具，调用格式是 `<tool_call>{...}</tool_call>`，最终答案放进 `<answer>`”。内容见 `DEFAULT_SYSTEM_PROMPT`（[:30](../SFT/data_clean/prepare_trajectories.py#L30)）。 |
| `user` | 第一条是用户的**问题**；之后的 `user` 都是**工具返回的结果**，用 `<tool_response>` 包起来。工具结果不是模型生成的，所以按惯例放在 user 这一侧。 |
| `assistant` | 模型的输出。中间轮是 `<think>思考</think>` + `<tool_call>调用工具</tool_call>`；最后一轮是 `<think>思考</think>` + `<answer>最终答案</answer>`。 |

SFT 只对 `assistant` 的 token 计算损失，也就是让模型学习：看到问题和工具结果之后，怎么思考、怎么调用工具、怎么作答。

### 完整示例：输入

下面这条原始 REDSearcher 轨迹是为了说明而构造的，每种情况都对应代码里的一个分支：

```python
row = {
  "meta": {"question": "DR-Venus 论文是哪一年在 arXiv 上发布的？", "answer": "2026"},
  "messages": [
    # [0] 问题
    {"role": "user", "content": "DR-Venus 论文是哪一年在 arXiv 上发布的？"},
    # [1] JSON 不规范：单引号、key 不带引号、用的是 queries、值是字符串
    {"role": "assistant", "content": "<think>我需要先搜索这篇论文</think><tool_call>{name:'search', arguments:{queries:'DR-Venus arXiv'}}</tool_call>"},
    # [2] Google 搜索结果是 markdown 文本
    {"role": "user", "content": "<tool_response>A Google search for 'DR-Venus arXiv' found 10 results:\n## Web Results\n1. [DR-Venus: Towards Frontier Edge-Scale ...](https://arxiv.org/abs/2604.19859)\nDate published: 2026-04-22\nWe present DR-Venus, a 4B deep research agent...</tool_response>"},
    # [3] 坏轮：没有 <think>
    {"role": "assistant", "content": "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":[\"DR-Venus\"]}}</tool_call>"},
    # [4] [3] 对应的工具返回
    {"role": "user", "content": "<tool_response>A Google search for 'DR-Venus' found ...</tool_response>"},
    # [5] visit：用的是 urls、值是字符串，goal 是 list
    {"role": "assistant", "content": "<think>打开 arXiv 页面确认日期</think><tool_call>{\"name\":\"visit\",\"arguments\":{\"urls\":\"https://arxiv.org/abs/2604.19859\",\"goal\":[\"查找提交日期\"]}}</tool_call>"},
    # [6] visit 返回的是摘要文本，不是搜索结果
    {"role": "user", "content": "<tool_response>The paper was submitted on 22 Apr 2026.</tool_response>"},
    # [7] 最后一轮：think 外面还有 <summary>
    {"role": "assistant", "content": "<think>两处来源都显示是 2026 年 4 月</think>\n<summary>arXiv 编号 2604 表示 2026 年 4 月</summary>\n<answer>2026</answer>"},
  ],
}
```

注意：原始数据里**没有 system 消息**，工具参数的写法也不统一。

### 逐条处理过程

`normalize_raw_messages`（[:402-447](../SFT/data_clean/prepare_trajectories.py#L402-L447)）一开始就把 system 放进结果（[:407](../SFT/data_clean/prepare_trajectories.py#L407)）：

```python
new_messages = [{"role": "system", "content": system_prompt}]
```

`system_prompt` 由 `choose_system_prompt`（[:112](../SFT/data_clean/prepare_trajectories.py#L112)）决定：

- 默认 `--system-prompt-mode template`：使用 `DEFAULT_SYSTEM_PROMPT`；如果传了 `--current-date`，末尾会加上 `Current date: 2026-03-01`。
- `--system-prompt-mode row`：优先使用这一行自带的 `system_prompt` 列，没有时再退回模板。

这样做的目的是让训练时的 system prompt 和推理时完全一致，否则模型学到的工具格式可能与推理环境对不上。

接下来遍历原始的 8 条消息：

| 原始下标 | 进入的分支 | 处理结果 |
|---|---|---|
| **[0]** | `index == 0`（[:416](../SFT/data_clean/prepare_trajectories.py#L416)） | 原样作为 `user` 问题加入。 |
| **[1]** | `role == "assistant"` → `clean_toolcall_output`（[:332](../SFT/data_clean/prepare_trajectories.py#L332)） | ① 正则取出 `<think>` 的内容；② 取出 `<tool_call>` 的内容，用 `json5` 解析（标准 `json` 解析不了单引号）；③ `normalize_tool_call`（[:290](../SFT/data_clean/prepare_trajectories.py#L290)）把 `queries` 改成 `query`，并把字符串包成 list；④ 按标准格式重新拼接。**保留**。 |
| **[2]** | `role == "user"`，前一条是 assistant | 先去掉原有的 `<tool_response>` 标签，发现包含 `Web Results`，于是调用 `parse_search_results`（[:273](../SFT/data_clean/prepare_trajectories.py#L273)）把 markdown 解析成 JSON：从标题行 `A Google search for '...' found N results:` 取出 query（去掉引号）；从 `1. [标题](url)` 取出 title 和 url；去掉 `Date published:` 行，剩下的作为 snippet。最后重新包上 `<tool_response>`。**保留**。 |
| **[3]** | `role == "assistant"` → `clean_toolcall_output` | 找不到 `<think>`，返回 `None`，代码执行 `continue`（[:440-441](../SFT/data_clean/prepare_trajectories.py#L440-L441)）。**只跳过这一轮**，整条轨迹保留。 |
| **[4]** | `role == "user"`，但前一条（`new_messages[-1]`）是 [2] 那条 user | 前一条不是 assistant，说明对应的调用已经被丢掉，这条返回也一起跳过（[:428-429](../SFT/data_clean/prepare_trajectories.py#L428-L429)）。如果不跳过，就会出现两条 user 连在一起。 |
| **[5]** | `role == "assistant"` | `normalize_tool_call` 的 visit 分支把 `urls` 改成 `url` 并包成 list，`goal` 从 list 中取第一个元素变成字符串。**保留**。 |
| **[6]** | `role == "user"`，前一条是 assistant | 内容里没有 `Web Results`，不做解析，只重新包上 `<tool_response>`。**保留**。 |
| **[7]** | `index == len-1`（[:420](../SFT/data_clean/prepare_trajectories.py#L420)）→ `clean_answer_output`（[:369](../SFT/data_clean/prepare_trajectories.py#L369)） | 取**最后一个** `<answer>`；把 `</think>` 之后的 `<summary>` 内容合并进 think。如果最后一条没有 `<answer>` 或 `<think>`，返回 `None`，**整条轨迹丢弃**（[:422-423](../SFT/data_clean/prepare_trajectories.py#L422-L423)）。 |

最后用 `deep_fix` 修复字符串中非法的 UTF-16 代理字符（[:447](../SFT/data_clean/prepare_trajectories.py#L447)）。外层的 `normalize_row`（[:484](../SFT/data_clean/prepare_trajectories.py#L484)）再从 `meta` 中取出 `question` 和 `gt`，组成一行输出。

### 完整示例：输出

```python
{
  "question": "DR-Venus 论文是哪一年在 arXiv 上发布的？",
  "gt": "2026",
  "messages": [
    {"role": "system",    "content": "You are a deep research assistant. ... # Tools ... <tools>{search...}{visit...}</tools> ..."},
    {"role": "user",      "content": "DR-Venus 论文是哪一年在 arXiv 上发布的？"},
    {"role": "assistant", "content": "<think>\n我需要先搜索这篇论文\n</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": [\"DR-Venus arXiv\"]}}\n</tool_call>"},
    {"role": "user",      "content": "<tool_response>\n[{\"query\": \"DR-Venus arXiv\", \"web_page_info_list\": [{\"title\": \"DR-Venus: Towards Frontier Edge-Scale ...\", \"url\": \"https://arxiv.org/abs/2604.19859\", \"snippet\": \"We present DR-Venus, a 4B deep research agent...\"}]}]\n</tool_response>"},
    {"role": "assistant", "content": "<think>\n打开 arXiv 页面确认日期\n</think>\n<tool_call>\n{\"name\": \"visit\", \"arguments\": {\"url\": [\"https://arxiv.org/abs/2604.19859\"], \"goal\": \"查找提交日期\"}}\n</tool_call>"},
    {"role": "user",      "content": "<tool_response>\nThe paper was submitted on 22 Apr 2026.\n</tool_response>"},
    {"role": "assistant", "content": "<think>\n两处来源都显示是 2026 年 4 月\n\narXiv 编号 2604 表示 2026 年 4 月\n</think>\n<answer>\n2026\n</answer>"},
  ],
}
```

与输入相比：

- **多了 system**，内容是统一的提示词和工具定义。
- 8 条消息变成 7 条：[3] 和 [4] 被成对去掉，角色严格按 `system → user → assistant → user → … → assistant` 交替。
- 工具调用统一成标准 JSON：`{"query": [...]}` 和 `{"url": [...], "goal": "..."}`，和 system prompt 中声明的参数 schema 一致。
- 搜索结果从 markdown 变成结构化 JSON。
- 最后一轮的 `<summary>` 被合并进 `<think>`，只剩 `<think>` + `<answer>` 两个标签。

### 实现要点

入口：`normalize_row`（[:484](../SFT/data_clean/prepare_trajectories.py#L484)）。如果数据带 `meta`，走 `normalize_raw_messages`（[:402](../SFT/data_clean/prepare_trajectories.py#L402)）；否则视为已清洗数据，走 `normalize_clean_messages`（只补上或替换 system prompt）。

> 注意：第一条和最后一条只按**下标**判断，第 0 条一律当作问题，最后一条一律当作最终答案，不看它们的 `role`。中间消息才按 `role` 分支处理。如果中间出现 `user`/`assistant` 以外的 role（比如 `tool`），会走到 [:445](../SFT/data_clean/prepare_trajectories.py#L445) 的 `return None`，整条轨迹丢弃。也就是说，这个函数默认原始数据中的工具返回已经用 `role="user"` 表示。

汇总如下：

| 消息位置 | 处理 | 失败时 |
|---|---|---|
| 开头 | 插入统一的 system prompt（`DEFAULT_SYSTEM_PROMPT`，可用 `--current-date` 加上日期；`--system-prompt-mode row` 则保留原轨迹的 prompt） | — |
| 第一条 | 作为 user 问题 | — |
| 最后一条 | `clean_answer_output`（[:369](../SFT/data_clean/prepare_trajectories.py#L369)）：取最后一个 `<answer>`；去掉 think 中残留的 `<answer>`/`<summary>` 标签；把 think 外的 `<summary>` 合并进 think | **整条丢弃**（`raw_clean_failed`） |
| 中间 assistant | `clean_toolcall_output`（[:332](../SFT/data_clean/prepare_trajectories.py#L332)）：用 `json5` 解析不规范的 JSON（单引号、key 不带引号）；再由 `normalize_tool_call`（[:290](../SFT/data_clean/prepare_trajectories.py#L290)）统一参数：`queries→query`、`urls→url`、字符串包成 list、list 形式的 `goal` 取第一个 | **只跳过这一轮** |
| 中间 user（工具返回） | 如果前一条不是 assistant（对应的 assistant 刚被跳过），也一起跳过，避免留下孤立的返回；含 `Web Results` 时用 `parse_search_results`（[:273](../SFT/data_clean/prepare_trajectories.py#L273)）解析成 `{query, web_page_info_list:[{title,url,snippet}]}` 结构，并去掉 `Date published:` 等元数据行 | — |

最后用 `deep_fix` 修复 UTF-16 代理字符，避免写 parquet 或 tokenize 时报错。

缺少 `question` 或 `gt` 的轨迹直接丢弃（`missing_question_or_gt`）。`gt` 字段会一直保留，供后续正确性过滤使用。

---

## ② 非法工具裁剪（Prune disallowed tools）

### 作用

目标模型推理时只有 `search` 和 `visit` 两个工具。原始轨迹里还有 `PythonInterpreter` 等其他工具的调用，如果保留，模型会学着调用一个推理环境中不存在的工具。

### 示例

```
[2] assistant: <tool_call>{"name":"PythonInterpreter",...}</tool_call>   ← 删除
[3] user:      <tool_response>计算结果 42</tool_response>                 ← 连带删除
[4] assistant: <tool_call>{"name":"search",...}</tool_call>               ← 保留
```

### 实现

`prune_disallowed_tool_turns`（[:541-587](../SFT/data_clean/prepare_trajectories.py#L541-L587)）：

- 对每条 assistant 消息，用 `extract_tool_names` 取出所有工具名，再与 `allowed_tools`（默认 `search visit`，可用 `--allowed-tools` 修改，不区分大小写）比较。
- 只要有一个工具不在允许列表中，就删除这条 assistant，并用内层 `while` 把紧跟在后面的所有 `<tool_response>` user 消息一起删除，保证调用和返回成对删除。
- 统计结果写入 `stats.json` 的 `pruned_tool_turns_removed`、`pruned_tool_response_turns_removed`、`pruned_tool_name_counts`。

论文统计：涉及 1,064 条轨迹、3,378 次调用。

---

## ③ 重复调用去重（Dedup）

### 作用

教师模型经常重复搜索同一个 query、重复访问同一个 URL。这些冗余轮次会让模型学到原地打转的行为，同时浪费上下文长度（长轨迹本来就接近 200K 的上限）。

### 示例

```
[2]  search ["DR-Venus", "Venus agent"]   → 保留，记录签名
[6]  visit  ["https://a.com"]             → 保留，记录签名
[10] search ["Venus agent", "DR-Venus"]   → 签名与 [2] 相同（只是顺序不同）→ 连同返回一起删除
[14] search ["DR-Venus", "RL training"]   → 签名是新的 → 保留
[18] visit  {"urls": "https://a.com"}     → 换了 key 的写法，签名仍与 [6] 相同 → 删除
```

### 实现

**签名**：`canonical_signature`（[:590-615](../SFT/data_clean/prepare_trajectories.py#L590-L615)）把一次调用映射成 `(工具名, 排序去重后的参数元组)`：

- `search`：取 `query`，没有时取 `queries`。
- `visit`：从 `url/urls/uri/uris/link/links/page/pages/query/queries` 中收集所有值。
- 用 `normalize_str_list` 去掉首尾空白和空值，再执行 `sorted(set(...))`。因此签名**与参数顺序、重复项、key 的写法无关**。
- 其他工具返回 `None`，不参与去重（经过 ② 之后实际只剩 search/visit）。

**去重**：`remove_duplicate_tool_calls`（[:618-663](../SFT/data_clean/prepare_trajectories.py#L618-L663)）：

- 用 `seen_signatures` 记录整条轨迹中已出现的签名。
- 一个 assistant 轮内可能有多个调用，**只有全部签名都已出现过时**（`all(...)`）才删除这一轮；只要有一个新签名就保留，并把这一轮的所有签名加入 `seen_signatures`。
- 删除方式与 ② 相同：assistant 和紧跟其后的 `<tool_response>` 一起删除，删除位置记录在 `duplicate_info` 中。
- 签名是精确匹配：`"DR-Venus"` 和 `"dr-venus"`、带不带结尾 `/` 的 URL 都算不同的签名。

论文统计：删除 15,728 次重复交互，主要是重复的 browse（即 `visit`）。

---

## ④ 结构校验（Validate）

### 作用

①–③ 删除若干轮之后，轨迹可能不完整，例如只剩 system 和 user，或者角色顺序被打乱。这一步做最终把关，保证每条样本都能被 chat template 正确渲染，并且 SFT 只对 assistant token 计算损失时 mask 能对齐。

### 示例（会被丢弃的情况）

| 情况 | 丢弃原因 |
|---|---|
| 去重后只剩 `[system, user]` | `too_short` |
| 中间某条 assistant 同时有 `<tool_call>` 和 `<answer>` | `assistant_tool_turn_contains_answer` |
| 最后一条 assistant 只有 `<tool_call>`，没有最终回答 | `final_assistant_missing_answer` |
| 第二条 user 没有 `<tool_response>` 标签 | `user_missing_tool_response_at_3` |

### 实现

`validate_messages` + `validate_assistant_message`（[:666-716](../SFT/data_clean/prepare_trajectories.py#L666-L716)）：

- 至少 3 条消息，第 0 条必须是 `system`。
- 之后奇数位置是 `user`，偶数位置是 `assistant`，严格交替。
- 第 1 条 user（问题）**不能**含 `<tool_response>`；之后每条 user **必须**含完整的 `<tool_response>...</tool_response>`。
- 中间 assistant：必须有 `<think>` 和 `<tool_call>`，不能有 `<answer>`。
- 最后一条 assistant：必须有 `<think>` 和 `<answer>`，不能有 `<tool_call>`。
- 不通过的整条丢弃，原因计入 `stats.json` 的 `drop_reasons`。

---

## （外部）LLM-judge 正确性过滤：10,001 → 9,365

### 作用

格式正确不代表答案正确。这一步用 LLM 判断最终 `<answer>` 是否与 `gt` 一致，只保留答对的轨迹，避免模型模仿错误的推理过程。

### 实现

**不在本仓库中**。论文用 Qwen3-235B-A22B-Instruct-2507 作为 judge（实现参考 Tongyi-DeepResearch），保留 9,365 条，占有效轨迹的 93.65%。本脚本只负责在 ① 中保留 `gt` 字段（`extract_question_gt`，[:472](../SFT/data_clean/prepare_trajectories.py#L472)）。

如果要严格复现论文，需要在 ④ 之后、⑤ 之前自行加一步判定，可以先用 `--keep-intermediate` 导出 `stage2_deduped.parquet`，过滤后再作为输入重新运行。已清洗格式的输入（`question/messages/gt`）走 `normalize_clean_messages`，不会被重复处理。RL 阶段的 `RL/verl/utils/reward_score/llm_judge.py` 思路相同，可以参考，但它用于计算训练奖励，不是数据过滤。

---

## ⑤ 按轮数重采样：9,365 → 18,745

### 作用

长程轨迹（几十到上百轮工具调用）最能体现深度研究能力，但数量较少。如果不上采样，SFT 会偏向“搜几次就回答”的短轨迹。重采样提高长轨迹的权重，让模型学会持续检索和交叉验证。

### 示例

| 轨迹轮数 | 倍数 | 输出份数 |
|---|---|---|
| 30 轮 | 1× | 1 |
| 80 轮 | 2× | 2 |
| 120 轮 | 5× | 5 |

### 实现

代码中的“采样权重”就是**把长轨迹原样复制几份**，没有使用加权采样器。实现分三步。

**1. 计算轮数**：`compute_turns`（[:719-720](../SFT/data_clean/prepare_trajectories.py#L719-L720)）

```python
return max(len(messages) - 1, 0) // 2
```

去掉第一条 system 后，每一对 `user + assistant` 算一轮。以 ① 节的示例为例：7 条消息 → (7−1)//2 = **3 轮**，正好等于 assistant 消息的条数，也就是工具调用次数 + 1（最后回答的那一轮）。

轮数是在 ② 裁剪、③ 去重**之后**计算的（[:845](../SFT/data_clean/prepare_trajectories.py#L845)），所以被删除的重复调用不计入轮数，长轨迹不会因为原地打转而被虚高地上采样。

**2. 根据轮数确定倍数**：`compute_multiplier`（[:723-728](../SFT/data_clean/prepare_trajectories.py#L723-L728)）

```python
if turns <= short_max:  return short_mult    # ≤50    → 1
if turns <= medium_max: return medium_mult   # 51–100 → 2
return long_mult                             # >100   → 5
```

阈值和倍数可通过 `--short-max-turns`、`--medium-max-turns`、`--short-multiplier`、`--medium-multiplier`、`--long-multiplier` 修改，默认值与论文一致（[:755-759](../SFT/data_clean/prepare_trajectories.py#L755-L759)）。

**3. 复制并打乱**：`main` 中的重采样循环（[:880-910](../SFT/data_clean/prepare_trajectories.py#L880-L910)）

```python
for sample_copy in range(multiplier):
    new_record = dict(record)
    new_record["_resample_multiplier"] = multiplier
    new_record["_sample_copy"] = sample_copy
    resampled_rows.append(new_record)
...
if args.shuffle:
    final_df = final_df.sample(frac=1.0, random_state=args.seed)
```

- 每条轨迹复制 `multiplier` 份，副本内容完全相同。同一个循环里还会统计 `original_bucket_counts`、`resampled_bucket_counts` 和 `multiplier_counts`，写入 `stats.json`。
- 加 `--shuffle` 时用 `--seed` 整体打乱，避免同一条轨迹的副本连在一起出现。
- 实际效果：一条轨迹在一个 epoch 中被训练 1、2 或 5 次，对梯度的贡献也相应放大。
- 默认只输出 `question / messages / gt` 三列；加 `--keep-metadata` 会额外保留 `_turns`、`_resample_multiplier`、`_sample_copy` 等调试列，可以按条查看每条轨迹是第几份副本。

### 用论文数字验算

用 N = 9,365 和重采样前的两个占比（>50 轮占 60.28%，>100 轮占 13.29%），可以反推出各区间的条数：

| 区间 | 重采样前 | 倍数 | 重采样后 |
|---|---|---|---|
| 0–50 轮 | 9,365 − 5,645 = **3,720** | 1× | 3,720 |
| 51–100 轮 | 5,645 − 1,245 = **4,400** | 2× | 8,800 |
| >100 轮 | 9,365 × 13.29% ≈ **1,245** | 5× | 6,225 |
| 合计 | 9,365 | | **18,745** ✓ |

其中 >50 轮的 5,645 条由 9,365 × 60.28% 算出。

重采样后的占比：

- 超过 50 轮：(8,800 + 6,225) / 18,745 = **80.15%** ✓
- 超过 100 轮：6,225 / 18,745 = **33.21%** ✓

三个数字都与论文一致，说明论文结果就是用这段代码的默认参数得到的。

自己运行时，可以在 `stats.json` 中核对：`turn_buckets.original` / `turn_buckets.resampled` 是各区间在重采样前后的条数，`multiplier_distribution` 是各倍数对应的原始轨迹数。

---

## 输出与统计

| 文件 | 内容 | 开关 |
|---|---|---|
| `cleaned_resampled.parquet` | 最终训练数据 | 默认 |
| `stats.json` | 各阶段行数、`drop_reasons`、裁剪/去重计数、`turn_buckets`（重采样前后）、`multiplier_distribution`、各文件明细 | 默认 |
| `stage1_normalized.parquet` | ① 之后的数据 | `--keep-intermediate` |
| `stage2_deduped.parquet` | ②–④ 之后的数据 | `--keep-intermediate` |
| `cleaned_resampled.jsonl` | 最终数据的 JSONL 版本 | `--write-jsonl` |

## 小结

| 步骤 | 解决的问题 |
|---|---|
| ① 格式对齐 | 格式统一、可解析，和推理时的格式一致 |
| ② 非法工具裁剪 | 不学调用推理时不存在的工具 |
| ③ 重复调用去重 | 不学无效的重复探索，节省上下文 |
| ④ 结构校验 | 保证样本结构合法、损失 mask 正确 |
| LLM-judge 过滤 | 只学答对的轨迹 |
| ⑤ 重采样 | 提高长程推理轨迹的训练权重 |
