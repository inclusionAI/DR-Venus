# Failure taxonomy

Use one primary label and optional secondary labels per failed trajectory.

| Label | Definition | Evidence to save |
|---|---|---|
| retrieval_miss | Search never surfaces a relevant source | queries and search result IDs |
| shallow_search | Agent repeats snippets without opening evidence | search/visit counts and browse ratio |
| bad_page | Page unavailable, blocked, stale, or irrelevant | URL, status, cache metadata |
| summarizer_loss | Relevant evidence is dropped or distorted by summarization | raw page hash and summary |
| reasoning_error | Evidence is present but conclusion is wrong | cited evidence and final reasoning |
| unsupported_answer | Answer lacks adequate retrieved support | final answer and visited sources |
| format_error | Malformed think/tool/answer protocol | raw assistant turn |
| duplicate_loop | Repeated query, URL, or reasoning tail | dedup/retry trace |
| timeout_or_limit | Tool timeout, time budget, context, or turn cap ends run | termination reason and budget |
| judge_error | Automated verdict disagrees with calibrated human review | judge prompt/output and reviewer label |
| reward_hacking | Policy improves reward without improving task behavior | reward/trajectory comparison |
| train_eval_leakage | Query or near-duplicate appears across splits | normalized IDs and similarity evidence |
