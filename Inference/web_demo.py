#!/usr/bin/env python
# coding: utf-8
"""
DR-Venus Web Demo with Gradio
用户可以通过Web界面配置API并实时查看Research Process
"""

import argparse
import json
import logging
import os
import random
import re
import time
import traceback
from threading import Lock
from typing import Callable, Optional

import gradio as gr
import openai
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizer

from tool_server.execute_tools import custom_call_tool

# 加载环境变量
load_dotenv()


# Suppress verbose library logging
for _lib in ("openai", "httpx", "urllib3", "requests", "filelock", "httpcore"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For each user request, you must actively seek out and **cross-check information** from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer in `<answer></answer>` tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: 2026-03-01
"""

ALLOWED_ARGS = {
    "search": {"query"},
    "visit": {"url", "goal"},
}

ALLOWED_TOOLS = set(ALLOWED_ARGS.keys())


def _filter_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}
    allowed = ALLOWED_ARGS.get(tool_name, set())
    return {k: v for k, v in args.items() if k in allowed}


class ResearchProblemSolver:
    """支持实时回调的推理求解器"""

    def __init__(
        self,
        question: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        openai_client: openai.Client,
        model_name: str,
        max_steps: int,
        time_limit: int = 9000,
        verbose: bool = False,
        max_retries: int = 5,
        callback: Optional[Callable] = None,
    ):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        self.question = question
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.openai_client = openai_client
        self.model_name = model_name
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max(max_retries, 1)
        self.callback = callback  # 实时回调函数

        self.start_time = time.time()
        self.turns = 0
        self.parse_error_count = 0
        self.tool_call_count = 0
        self.dedup_count = 0
        self._tool_call_history: dict[str, bool] = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"[Solver] {msg}")

    @staticmethod
    def _tool_call_key(tool_call: dict) -> str:
        return json.dumps(
            {"name": tool_call["name"], "arguments": tool_call["arguments"]},
            sort_keys=True, ensure_ascii=False,
        )

    @staticmethod
    def _is_repetitive(content: str, tail_len: int = 50, threshold: int = 5) -> bool:
        if not content or len(content) < tail_len:
            return False
        tail = content[-tail_len:]
        return content.count(tail) > threshold

    def count_tokens(self) -> int:
        rendered = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return len(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )

    def _call_llm_once(self) -> str | None:
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                stop=["<tool_response>", "\n<tool_response>"],
                temperature=1.0,
                max_tokens=15000,
                top_p=0.95,
                presence_penalty=1.1,
                extra_body={"top_k": 20},
            )
            content = response.choices[0].message.content
            return content.strip() if content and content.strip() else None
        except Exception as e:
            self.log(f"API error: {e}")
            return None

    def parse_llm_response(self, content: str, step: int) -> tuple[int, str, object]:
        """解析LLM响应"""
        THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        self.log(f"step={step}, content_len={len(content)}")

        think_match = THINK_RE.search(content)
        if not think_match:
            return 2, "Missing <think></think>", ""

        thinking = think_match.group(1).strip()
        answer_match = ANSWER_RE.search(content)
        tool_match = TOOL_RE.search(content)

        if answer_match and not tool_match:
            return 0, thinking, answer_match.group(1).strip()

        if tool_match and not answer_match:
            raw_tool_json = tool_match.group(1).strip()
            try:
                import json5
                tool_call = json5.loads(raw_tool_json)

                if not isinstance(tool_call, dict):
                    return 2, f"Tool call should be dict, got {type(tool_call)}", ""
                if "name" not in tool_call or "arguments" not in tool_call:
                    return 2, f"Tool call missing fields, raw={raw_tool_json}", ""

                if tool_call["name"] not in ALLOWED_TOOLS:
                    return 2, f"Unknown tool '{tool_call['name']}'. Available: {', '.join(sorted(ALLOWED_TOOLS))}", ""

                tool_call["arguments"] = _filter_args(tool_call["name"], tool_call["arguments"])
                return 1, thinking, tool_call

            except Exception as e:
                return 2, f"Tool call parse error: {repr(e)}, raw={raw_tool_json}", ""

        return 2, "Ambiguous or incomplete output", ""

    def execute_tool(self, tool_call: dict) -> str:
        try:
            results = custom_call_tool(tool_call)
            if results and len(results) > 0:
                return results
            return "ERROR: empty tool result"
        except Exception as e:
            return f"ERROR: tool execution failed: {e}"

    def extract_final_answer(self) -> str | None:
        ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    return None
                stripped = THINK_RE.sub("", content).strip()
                m = ANSWER_RE.search(stripped)
                if m and m.group(1).strip():
                    return m.group(1).strip()
                return content.strip()
        return None

    def _make_result(self, termination: str, prediction: str | None = None) -> dict:
        if prediction is None:
            prediction = self.extract_final_answer() or "No answer found"
        return {
            "prediction": prediction,
            "termination": termination,
            "turns": self.turns,
            "parse_error_count": self.parse_error_count,
            "tool_call_count": self.tool_call_count,
            "dedup_count": self.dedup_count,
            "final_token_count": self.count_tokens(),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
        }

    @staticmethod
    def _sanitize_content(content: str) -> str:
        if "<tool_response>" in content:
            content = content[: content.find("<tool_response>")]
        if "<think>" in content and "</think>" not in content:
            for tag in ("<answer>", "<tool_call>"):
                pos = content.find(tag)
                if pos != -1:
                    content = content[:pos] + "\n</think>\n" + content[pos:]
                    break
            else:
                content = content + "\n</think>"
        if "<think>" not in content and "</think>" in content:
            content = "<think>\n" + content
        return content

    def _generate(self, step: int) -> tuple[str | None, int, str, object]:
        last_content = None
        last_status = -1
        last_thinking = ""
        last_payload = None

        for attempt in range(self.max_retries):
            is_last = (attempt == self.max_retries - 1)
            backoff = min(2 ** attempt + random.uniform(0, 1), 30)

            content = self._call_llm_once()
            if not content:
                self.log(f"[_generate] API failure (attempt {attempt + 1}/{self.max_retries})")
                if not is_last:
                    time.sleep(backoff)
                    continue
                return None, -1, "", None

            if self._is_repetitive(content):
                self.log(f"[_generate] Repeat detected (attempt {attempt + 1}/{self.max_retries})")
                last_content = content
                if not is_last:
                    time.sleep(backoff)
                    continue

            content = self._sanitize_content(content)
            status, thinking, payload = self.parse_llm_response(content, step)

            if status == 2:
                self.parse_error_count += 1
                self.log(f"[_generate] Parse error (attempt {attempt + 1}/{self.max_retries}): {thinking}")
                last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                if not is_last:
                    time.sleep(backoff)
                    continue
                return content, status, thinking, payload

            if status == 1:
                key = self._tool_call_key(payload)
                if key in self._tool_call_history:
                    self.dedup_count += 1
                    self.log(f"[_generate] Duplicate tool call (attempt {attempt + 1}/{self.max_retries}): {payload['name']}")
                    last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                    if not is_last:
                        time.sleep(backoff)
                        continue
                    return content, status, thinking, payload

            return content, status, thinking, payload

        if last_content is not None:
            return last_content, last_status, last_thinking, last_payload
        return None, -1, "", None

    def _force_answer(self, termination_ok: str, termination_fail: str) -> dict:
        content, status, thinking, payload = self._generate(step=-1)
        if content is None:
            return self._make_result(termination_fail)

        self.messages.append({"role": "assistant", "content": content})
        self.turns += 1

        if status == 0:
            return self._make_result(termination_ok, payload)
        return self._make_result(termination_fail)

    def solve(self) -> dict:
        """主循环，支持实时回调"""
        for step in range(self.max_steps):
            if time.time() - self.start_time > self.time_limit:
                return self._make_result("timeout")

            content, status, thinking, payload = self._generate(step)
            if content is None:
                return self._make_result("llm_error")

            self.messages.append({"role": "assistant", "content": content})

            # 回调：发送思考过程
            if self.callback:
                self.callback({
                    "step": self.turns + 1,
                    "type": "thinking",
                    "thinking": thinking,
                    "status": "in_progress" if status != 0 else "completed"
                })

            if status == 0:  # final answer
                self.turns += 1
                result = self._make_result("answered", payload)
                if self.callback:
                    self.callback({
                        "step": self.turns,
                        "type": "answer",
                        "answer": payload,
                        "status": "completed"
                    })
                return result

            if status == 1:  # tool call
                self.turns += 1
                self.tool_call_count += 1
                self._tool_call_history[self._tool_call_key(payload)] = True
                tool_name = payload["name"]
                tool_args = payload["arguments"]

                # 回调：发送工具调用信息
                if self.callback:
                    self.callback({
                        "step": self.turns,
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": tool_args,
                        "status": "executing"
                    })

                result = self.execute_tool(payload)

                # 回调：发送工具执行结果
                if self.callback:
                    self.callback({
                        "step": self.turns,
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": result[:50000] if len(result) > 50000 else result,  # 截断太长结果
                        "status": "completed"
                    })

                self.messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>",
                })

            if status == 2:  # parse error
                self.turns += 1
                self.log(f"Parse error at step {step}: {thinking}")
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"<tool_response>\nERROR!\n"
                        "Please call a tool with the correct format, or if you are ready to answer, provide the entire final answer in `<answer></answer>` tags.\n</tool_response>"
                    ),
                })
                if self.callback:
                    self.callback({
                        "step": self.turns,
                        "type": "error",
                        "error": thinking,
                        "status": "retrying"
                    })

            token_count = self.count_tokens()
            if token_count > self.max_len:
                self.messages[-1] = {
                    "role": "user",
                    "content": (
                        "<tool_response>\nYou have reached the maximum context length. "
                        "Based on the information and knowledge you have gathered, please provide the entire final answer in `<answer></answer>` tags now.\n</tool_response>"
                    ),
                }
                return self._force_answer("answered_token_limit", "token_limit_no_answer")

        self.messages[-1] = {
            "role": "user",
            "content": (
                "<tool_response>\nYou've reached the maximum number of tool calls. "
                "Based on the information and knowledge you have gathered, please provide the entire final answer in `<answer></answer>` tags now.\n</tool_response>"
            ),
        }
        return self._force_answer("answered_max_steps", "max_steps")


def build_openai_client(base_url: str, api_key: str = "123") -> openai.Client:
    """构建OpenAI客户端"""
    return openai.Client(
        api_key=api_key,
        base_url=base_url,
    )


def check_vllm_server(url: str) -> tuple[bool, str]:
    """检查vLLM服务器是否可用"""
    try:
        response = requests.get(f"{url}/v1/models", timeout=5)
        if response.status_code == 200:
            return True, "vLLM Server Connected"
        else:
            return False, f"Server Error: {response.status_code}"
    except Exception as e:
        return False, f"连接失败: {str(e)}"


# 全局状态
global_state = {
    "tokenizer": None,
    "current_solver": None,
    "is_running": False,
    "events": [],  # 存储推理事件
    "status": "idle",  # idle, running, completed, error
}


def load_tokenizer(model_path: str):
    """加载tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"加载tokenizer失败: {e}")


def research_task(
    question: str,
    model_path: str,
    vllm_url: str,
    serper_key: str,
    jina_key: str,
    summary_api_key: str,
    summary_api_base: str,
    summary_model_name: str,
    proxy: str,
    max_steps: int,
    progress: gr.Progress,
):
    """执行研究任务的主函数"""
    global global_state

    # 设置环境变量
    os.environ["SERPER_KEY_ID"] = serper_key
    os.environ["JINA_API_KEYS"] = jina_key
    os.environ["API_KEY"] = summary_api_key
    os.environ["API_BASE"] = summary_api_base
    os.environ["SUMMARY_MODEL_NAME"] = summary_model_name
    os.environ["PROXY"] = proxy

    # 构建OpenAI客户端
    vllm_url = vllm_url.rstrip("/")
    if not vllm_url.startswith("http"):
        vllm_url = f"http://{vllm_url}"
    # 确保URL以/v1结尾
    if not vllm_url.endswith("/v1"):
        vllm_url = vllm_url.rstrip("/v1") + "/v1"

    client = build_openai_client(vllm_url)

    # 加载tokenizer
    try:
        tokenizer = load_tokenizer(model_path)
    except Exception as e:
        yield {"step": 0, "type": "error", "message": f"加载tokenizer失败: {e}", "status": "error"}
        return

    # 回调函数：实时更新UI
    def callback(event):
        yield event

    try:
        solver = ResearchProblemSolver(
            question=question,
            tokenizer=tokenizer,
            max_len=240000,
            max_steps=max_steps,
            openai_client=client,
            model_name="model",
            time_limit=3600,
            verbose=False,
            max_retries=3,
            callback=callback,
        )

        global_state["current_solver"] = solver

        # 开始推理
        result = solver.solve()

        # 返回最终结果
        yield {
            "step": result["turns"],
            "type": "final",
            "answer": result["prediction"],
            "termination": result["termination"],
            "tool_call_count": result["tool_call_count"],
            "elapsed_seconds": result["elapsed_seconds"],
            "status": "completed"
        }

    except Exception as e:
        yield {
            "step": 0,
            "type": "error",
            "message": f"Research Error: {str(e)}\n{traceback.format_exc()}",
            "status": "error"
        }
    finally:
        global_state["is_running"] = False


def simple_markdown_to_html(text: str) -> str:
    """简单的Markdown转HTML"""
    if not text:
        return ""

    import re

    # 转义HTML特殊字符
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # 代码块 (先处理，避免其他转换影响)
    text = re.sub(r'```(\w*)\n?(.*?)```', r'<pre><code class="language-\1">\2</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # 标题
    text = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', text, flags=re.MULTILINE)
    text = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # 粗体和斜体
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

    # 链接
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # 列表
    text = re.sub(r'^\* (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    # 换行
    text = text.replace('\n\n', '</p><p>')
    text = '<p>' + text + '</p>'

    # 清理空段落
    text = text.replace('<p></p>', '')
    text = text.replace('<p><br>', '<p>')
    text = text.replace('<br></p>', '</p>')

    return text


def format_step_event(event: dict) -> str:
    """格式化步骤事件为可显示的HTML"""
    step = event.get("step", "?")
    event_type = event.get("type", "unknown")

    # 统一的卡片样式
    card_style = """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <div style="color: white; font-weight: bold; font-size: 14px; margin-bottom: 8px;">
            🔄 Step {step}
        </div>
    """

    if event_type == "thinking":
        thinking = event.get("thinking", "")
        # 使用Markdown转换
        thinking_html = simple_markdown_to_html(thinking)
        return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 24px; margin-right: 10px;">🧠</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Thinking</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 10px; padding: 16px; color: #333; font-size: 14px; line-height: 1.8; max-height: 400px; overflow-y: auto;">
        <div class="markdown-content">{thinking_html}</div>
    </div>
</div>
"""
    elif event_type == "tool_call":
        tool = event.get("tool", "")
        args = event.get("args", {})
        if tool == "search":
            queries = args.get("query", [])
            query_text = "<br>".join([f"• {q}" for q in queries])
            return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 24px; margin-right: 10px;">🔍</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Searching</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 10px; padding: 16px;">
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">🔎 Search Query:</div>
        <div style="color: #555; font-size: 14px; line-height: 1.8;">
            {query_text}
        </div>
    </div>
</div>
"""
        elif tool == "visit":
            urls = args.get("url", [])
            goal = args.get("goal", "")
            if isinstance(urls, str):
                urls = [urls]
            url_text = "<br>".join([f"• {u}" for u in urls])
            return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 24px; margin-right: 10px;">🌐</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Visiting Page</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 10px; padding: 16px;">
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">🎯 Visit Goal:</div>
        <div style="color: #555; font-size: 14px; margin-bottom: 12px; padding: 8px; background: #f8f9fa; border-radius: 6px;">
            {goal}
        </div>
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">📄 URL:</div>
        <div style="color: #0066cc; font-size: 13px; word-break: break-all;">
            {url_text}
        </div>
    </div>
</div>
"""
        else:
            return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
">
    <div style="color: white; font-weight: bold;">Step {step} - {tool}</div>
</div>
"""
    elif event_type == "tool_result":
        tool = event.get("tool", "")
        result = event.get("result", "")

        if tool == "search":
            try:
                search_data = json.loads(result)
                output = f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 24px; margin-right: 10px;">📋</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Search Results</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 12px; padding: 12px;">
"""
                # 收集所有URL
                all_urls = []
                for query_result in search_data:
                    pages = query_result.get("web_page_info_list", [])
                    for page in pages:
                        url = page.get("url", "")
                        if url:
                            all_urls.append(url)

                # 只显示URL列表
                for i, url in enumerate(all_urls, 1):
                    # 提取域名用于favicon
                    import re
                    domain_match = re.search(r'https?://([^/]+)', url)
                    domain = domain_match.group(1) if domain_match else ""
                    favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
                    output += f"""
        <div style="background: #f0f4f8; border-radius: 6px; padding: 8px 12px; margin: 4px 0; display: flex; align-items: center; gap: 10px;">
            <img src="{favicon_url}" style="width: 20px; height: 20px; border-radius: 4px;" onerror="this.style.display='none'">
            <span style="color: #0066cc; font-size: 13px; font-family: monospace; word-break: break-all;">{url}</span>
        </div>
"""
                output += "</div></div>"
                return output
            except Exception as e:
                # JSON解析失败，尝试直接解析原始文本
                try:
                    import re
                    match = re.search(r'\[.*\]', result, re.DOTALL)
                    if match:
                        search_data = json.loads(match.group(0))
                        # 简化显示：只显示URL列表
                        output = f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 24px; margin-right: 10px;">📋</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Search Results</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 12px; padding: 12px;">
"""
                        all_urls = []
                        for query_result in search_data:
                            pages = query_result.get("web_page_info_list", [])
                            for page in pages:
                                url = page.get("url", "")
                                if url:
                                    all_urls.append(url)

                        for i, url in enumerate(all_urls, 1):
                            import re
                            domain_match = re.search(r'https?://([^/]+)', url)
                            domain = domain_match.group(1) if domain_match else ""
                            favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
                            output += f"""
        <div style="background: #f0f4f8; border-radius: 6px; padding: 8px 12px; margin: 4px 0; display: flex; align-items: center; gap: 10px;">
            <img src="{favicon_url}" style="width: 20px; height: 20px; border-radius: 4px;" onerror="this.style.display='none'">
            <span style="color: #0066cc; font-size: 13px; font-family: monospace; word-break: break-all;">{url}</span>
        </div>
"""
                        output += "</div></div>"
                        return output
                except:
                    pass

                # 如果还是失败，尝试提取URL显示
                try:
                    import re
                    urls = re.findall(r'https?://[^\s"\'\]\)]+', result)
                    if urls:
                        output = f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 24px; margin-right: 10px;">📋</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Search Results</span>
    </div>
    <div style="background: rgba(255,255,255,0.95); border-radius: 12px; padding: 12px;">
"""
                        for i, url in enumerate(urls[:10], 1):  # 最多显示10个
                            domain_match = re.search(r'https?://([^/]+)', url)
                            domain = domain_match.group(1) if domain_match else ""
                            favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
                            output += f"""
        <div style="background: #f0f4f8; border-radius: 6px; padding: 8px 12px; margin: 4px 0; display: flex; align-items: center; gap: 10px;">
            <img src="{favicon_url}" style="width: 20px; height: 20px; border-radius: 4px;" onerror="this.style.display='none'">
            <span style="color: #0066cc; font-size: 13px; font-family: monospace; word-break: break-all;">{url}</span>
        </div>
"""
                        output += "</div></div>"
                        return output
                except:
                    pass

                # 如果完全失败，显示原始数据
                return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 24px; margin-right: 10px;">⚠️</span>
        <span style="color: white; font-weight: bold;">Step {step} - Search Results (Data Truncated)</span>
    </div>
    <div style="background: rgba(255,255,255,0.9); border-radius: 10px; padding: 12px; max-height: 300px; overflow-y: auto;">
        <div style="color: #666; font-size: 13px; word-break: break-all; white-space: pre-wrap;">{result}</div>
    </div>
</div>
"""

        elif tool == "visit":
            # 格式化visit结果
            try:
                if result.startswith("The useful information"):
                    # 已经是格式化文本，提取URL和摘要
                    lines = result.split('\n')
                    url_line = ""
                    summary_lines = []
                    in_summary = False
                    for line in lines:
                        if "The useful information in" in line:
                            # 提取URL
                            import re
                            match = re.search(r'https?://[^\s]+', line)
                            if match:
                                url_line = match.group(0)
                        if "Summary:" in line:
                            in_summary = True
                        if in_summary and line.strip():
                            summary_lines.append(line.strip())

                    summary_text = " ".join(summary_lines[:5])
                    return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 24px; margin-right: 10px;">📄</span>
        <span style="color: #333; font-weight: bold; font-size: 16px;">Step {step} - Page Visit Result</span>
    </div>
    <div style="background: white; border-radius: 12px; padding: 16px;">
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">🔗 URL:</div>
        <div style="color: #0066cc; font-size: 13px; word-break: break-all; background: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-family: monospace;">
            {url_line}
        </div>
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">📝 LLM Summary:</div>
        <div style="color: #555; font-size: 14px; line-height: 1.7; background: #f8f9fa; padding: 12px; border-radius: 8px; max-height: 200px; overflow-y: auto;">
            {summary_text}
        </div>
    </div>
</div>
"""
                else:
                    # 尝试解析JSON
                    visit_data = json.loads(result)
                    output = f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 24px; margin-right: 10px;">📄</span>
        <span style="color: #333; font-weight: bold; font-size: 16px;">Step {step} - Page Visit Result</span>
    </div>
"""
                    if isinstance(visit_data, list):
                        for i, item in enumerate(visit_data, 1):
                            url = item.get("url", "")
                            rational = item.get("rational", "")
                            evidence = item.get("evidence", "")[:300]
                            summary = item.get("summary", "")
                            output += f"""
    <div style="background: white; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <div style="color: #333; font-weight: 600; margin-bottom: 8px;">🔗 URL {i}:</div>
        <div style="color: #0066cc; font-size: 13px; word-break: break-all; background: #f8f9fa; padding: 8px; border-radius: 6px; margin-bottom: 10px; font-family: monospace;">
            {url}
        </div>
        <div style="color: #333; font-weight: 600; margin-bottom: 6px;">📝 Summary:</div>
        <div style="color: #555; font-size: 13px; line-height: 1.6;">{summary}</div>
    </div>
"""
                    output += "</div>"
                    return output
            except:
                # 简单截断显示
                return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
">
    <div style="color: white; font-weight: bold;">Step {step} - Visit Result</div>
    <div style="color: white; font-size: 13px; margin-top: 8px;">{result[:500]}...</div>
</div>
"""

        else:
            return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
">
    <div style="color: white; font-weight: bold;">Step {step} - {tool}</div>
</div>
"""

    elif event_type == "answer":
        answer = event.get("answer", "")
        # 转换Markdown
        answer_html = simple_markdown_to_html(answer)
        return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
">
    <div style="display: flex; align-items: center; margin-bottom: 12px;">
        <span style="font-size: 24px; margin-right: 10px;">🎯</span>
        <span style="color: white; font-weight: bold; font-size: 16px;">Step {step} - Final Answer</span>
    </div>
    <div class="markdown-content" style="background: rgba(255,255,255,0.95); border-radius: 12px; padding: 20px; color: #333; font-size: 15px; line-height: 1.8; max-height: 400px; overflow-y: auto;">
        {answer_html}
    </div>
</div>
"""

    elif event_type == "error":
        error = event.get("error", "")
        message = event.get("message", "")
        return f"""
<div class="step-card" style="
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 16px 0;
">
    <div style="display: flex; align-items: center;">
        <span style="font-size: 24px; margin-right: 10px;">❌</span>
        <span style="color: white; font-weight: bold;">Step {step} - Error</span>
    </div>
    <div style="color: white; margin-top: 8px; font-size: 14px;">{error or message}</div>
</div>
"""
    elif event_type == "final":
        answer = event.get("answer", "")
        termination = event.get("termination", "")
        tool_count = event.get("tool_call_count", 0)
        elapsed = event.get("elapsed_seconds", 0)
        turns = event.get("turns", 0)
        # 转换Markdown
        answer_html = simple_markdown_to_html(answer)
        return f"""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
">
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="font-size: 36px; margin-right: 12px;">🎉</span>
        <span style="color: white; font-size: 24px; font-weight: bold;">Research Complete!</span>
    </div>
    <div style="display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;">
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 12px; text-align: center; min-width: 80px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Status</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{termination}</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 12px; text-align: center; min-width: 80px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Tool Calls</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{tool_count}次</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 12px; text-align: center; min-width: 80px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Turns</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{turns}轮</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 12px; text-align: center; min-width: 80px;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">Time</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{elapsed}秒</div>
        </div>
    </div>
    <div style="background: white; border-radius: 16px; padding: 24px; max-height: 500px; overflow-y: auto;">
        <div style="color: #1a1a1a; font-weight: 700; font-size: 18px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 3px solid #667eea; display: flex; align-items: center; gap: 8px;">
            <span>📝</span> Final Answer
        </div>
        <div class="final-answer">
            {answer_html}
        </div>
    </div>
</div>
"""

    return ""


# 全局自定义CSS样式
CUSTOM_CSS = """
/* 全局样式 */
.gradio-container {
    max-width: 1400px !important;
    min-width: 1200px !important;
    margin: auto !important;
}

.main {
    min-width: 1200px !important;
}

/* Markdown内容样式 */
.markdown-content, .final-answer, .thinking-content {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.8;
    font-size: 15px;
    color: #333;
}

.markdown-content h1, .final-answer h1 {
    font-size: 24px;
    font-weight: 700;
    margin: 16px 0 12px 0;
    color: #1a1a1a;
    border-bottom: 2px solid #667eea;
    padding-bottom: 8px;
}

.markdown-content h2, .final-answer h2 {
    font-size: 20px;
    font-weight: 600;
    margin: 14px 0 10px 0;
    color: #2d2d2d;
}

.markdown-content h3, .final-answer h3 {
    font-size: 17px;
    font-weight: 600;
    margin: 12px 0 8px 0;
    color: #404040;
}

.markdown-content p, .final-answer p {
    margin: 10px 0;
}

.markdown-content ul, .final-answer ul,
.markdown-content ol, .final-answer ol {
    margin: 10px 0;
    padding-left: 24px;
}

.markdown-content li, .final-answer li {
    margin: 6px 0;
    line-height: 1.6;
}

.markdown-content code, .final-answer code {
    background: #f5f5f5;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
    font-size: 13px;
    color: #e83e8c;
}

.markdown-content pre, .final-answer pre {
    background: #2d2d2d;
    color: #f8f8f2;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 12px 0;
}

.markdown-content pre code, .final-answer pre code {
    background: transparent;
    padding: 0;
    color: #f8f8f2;
}

.markdown-content blockquote, .final-answer blockquote {
    border-left: 4px solid #667eea;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f8f9fa;
    color: #555;
}

.markdown-content a, .final-answer a {
    color: #667eea;
    text-decoration: none;
}

.markdown-content a:hover, .final-answer a:hover {
    text-decoration: underline;
}

.markdown-content strong, .final-answer strong {
    font-weight: 600;
    color: #1a1a1a;
}

.markdown-content em, .final-answer em {
    font-style: italic;
}

.markdown-content hr, .final-answer hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 16px 0;
}

.markdown-content table, .final-answer table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}

.markdown-content th, .final-answer th,
.markdown-content td, .final-answer td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

.markdown-content th, .final-answer th {
    background: #f5f5f5;
    font-weight: 600;
}

/* 滚动条美化 */
.markdown-content::-webkit-scrollbar, .final-answer::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.markdown-content::-webkit-scrollbar-track, .final-answer::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.markdown-content::-webkit-scrollbar-thumb, .final-answer::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
}

.markdown-content::-webkit-scrollbar-thumb:hover, .final-answer::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* 标题样式 */
.title-text {
    font-size: 28px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* 卡片容器 */
.card-container {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    padding: 20px;
"""


def gradio_app():
    """构建Gradio应用界面"""
    import threading

    with gr.Blocks(title="DR-Venus AI") as demo:
        gr.Markdown("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 32px;">🔍 DR-Venus AI</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 16px;">Deep Research AI System powered by Local LLM</p>
        </div>
        """, elem_id="header")

        # 状态存储
        process_output_state = gr.State(value="")
        final_answer_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Settings")

                model_path = gr.Textbox(
                    label="Model Path",
                    value="/home/changxiang/DR-Venus_model",
                    info="Local model path"
                )

                vllm_url = gr.Textbox(
                    label="vLLM Server URL",
                    value="http://127.0.0.1:6000",
                    info="vLLM API地址"
                )

                with gr.Row():
                    check_btn = gr.Button("Check Service", variant="secondary")
                    check_status = gr.Textbox(label="Service Status", interactive=False)

                check_btn.click(
                    fn=lambda url: check_vllm_server(url),
                    inputs=[vllm_url],
                    outputs=[check_status]
                )

                gr.Markdown("### 🔑 API Keys")

                serper_key = gr.Textbox(
                    label="Serper API Key (search)",
                    value=os.getenv("SERPER_KEY_ID", ""),
                    type="password"
                )

                jina_key = gr.Textbox(
                    label="Jina API Key (visit)",
                    value=os.getenv("JINA_API_KEYS", ""),
                    type="password"
                )

                gr.Markdown("### 📝 Summary Model Config")

                summary_api_key = gr.Textbox(
                    label="API Key",
                    value=os.getenv("API_KEY", ""),
                    type="password"
                )

                summary_api_base = gr.Textbox(
                    label="API Base URL",
                    value=os.getenv("API_BASE", "")
                )

                summary_model_name = gr.Textbox(
                    label="Model name",
                    value=os.getenv("SUMMARY_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")
                )

                proxy = gr.Textbox(
                    label="Proxy",
                    value=os.getenv("PROXY", ""),
                    info="optional"
                )

                max_steps = gr.Slider(
                    label="Max Steps",
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10
                )

            with gr.Column(scale=2):
                gr.Markdown("## 💬 Research Question")

                question_input = gr.Textbox(
                    label="Enter your research question",
                    placeholder="Enter your research question here...",
                    lines=3
                )

                with gr.Row():
                    start_btn = gr.Button("🚀 Start Research", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

                gr.Markdown("## 📊 Research Process")

                # 使用HTML组件来实时显示
                process_output = gr.HTML(
                    value="<p style='color: #666;'>*Waiting输入问题并点击Start Research...*</p>"
                )

                # Final Answer
                with gr.Accordion("📝 Final Answer", open=True):
                    final_answer = gr.HTML(
                        value="<p style='color: #666;'>*WaitingResearch Complete...*</p>"
                    )

                # 状态显示
                status_display = gr.Textbox(
                    label="Status",
                    value="空闲",
                    interactive=False
                )

        # 定义Start Research的函数（在线程中运行）
        def start_research_thread(question, model_path, vllm_url, serper_key, jina_key,
                                   summary_api_key, summary_api_base, summary_model_name,
                                   proxy, max_steps):
            global global_state

            # 重置状态
            global_state["events"] = []
            global_state["status"] = "running"
            global_state["is_running"] = True

            try:
                # 设置环境变量
                os.environ["SERPER_KEY_ID"] = serper_key
                os.environ["JINA_API_KEYS"] = jina_key
                os.environ["API_KEY"] = summary_api_key
                os.environ["API_BASE"] = summary_api_base
                os.environ["SUMMARY_MODEL_NAME"] = summary_model_name
                os.environ["PROXY"] = proxy

                # 规范化URL
                vllm_url = vllm_url.rstrip("/")
                if not vllm_url.startswith("http"):
                    vllm_url = f"http://{vllm_url}"
                if not vllm_url.endswith("/v1"):
                    vllm_url = vllm_url.rstrip("/v1") + "/v1"

                client = build_openai_client(vllm_url)

                # 加载tokenizer
                tokenizer = load_tokenizer(model_path)

                # 回调函数：保存事件到全局状态
                def callback(event):
                    global_state["events"].append(event)
                    return event

                solver = ResearchProblemSolver(
                    question=question,
                    tokenizer=tokenizer,
                    max_len=240000,
                    max_steps=max_steps,
                    openai_client=client,
                    model_name="model",
                    time_limit=3600,
                    verbose=False,
                    max_retries=3,
                    callback=callback,
                )

                global_state["current_solver"] = solver
                result = solver.solve()

                # 保存最终结果
                global_state["events"].append({
                    "type": "final",
                    "answer": result["prediction"],
                    "termination": result["termination"],
                    "tool_call_count": result["tool_call_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "turns": result["turns"]
                })
                global_state["status"] = "completed"

            except Exception as e:
                global_state["events"].append({
                    "type": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })
                global_state["status"] = "error"
            finally:
                global_state["is_running"] = False

        # 定义获取状态的函数
        def get_status():
            global global_state
            events = global_state.get("events", [])
            status = global_state.get("status", "idle")

            # 构建HTML
            html_parts = []
            for event in events:
                html_parts.append(format_step_event(event))

            process_html = "".join(html_parts)
            if not process_html:
                if status == "running":
                    process_html = """
<div style="text-align: center; padding: 40px;">
    <div style="font-size: 48px; margin-bottom: 16px;">🔄</div>
    <div style="color: #2196F3; font-size: 18px; font-weight: 600;">Research in Progress...</div>
    <div style="margin-top: 20px;">
        <div style="width: 200px; height: 4px; background: #e0e0e0; border-radius: 2px; margin: 0 auto;">
            <div style="width: 50%; height: 100%; background: linear-gradient(90deg, #4facfe, #00f2fe); border-radius: 2px; animation: loading 1.5s ease-in-out infinite;"></div>
        </div>
    </div>
</div>
<style>
@keyframes loading {
    0% { width: 0%; }
    50% { width: 70%; }
    100% { width: 100%; }
}
</style>
"""
                else:
                    process_html = """
<div style="text-align: center; padding: 60px; color: #999;">
    <div style="font-size: 64px; margin-bottom: 20px;">🔍</div>
    <div style="font-size: 20px; font-weight: 600; color: #666;">DR-Venus</div>
    <div style="margin-top: 12px; font-size: 14px;">输入Research Question，开始深度研究之旅</div>
</div>
"""

            # Final Answer
            final_html = ""
            for event in events:
                if event.get("type") == "final":
                    answer = event.get("answer", "")
                    termination = event.get("termination", "")
                    tool_count = event.get("tool_call_count", 0)
                    elapsed = event.get("elapsed_seconds", 0)
                    turns = event.get("turns", 0)
                    final_html = f"""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
">
    <div style="display: flex; align-items: center; margin-bottom: 16px;">
        <span style="font-size: 32px; margin-right: 12px;">🎉</span>
        <span style="color: white; font-size: 20px; font-weight: bold;">Research Complete!</span>
    </div>
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">状态</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{termination}</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">工具调用</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{tool_count}次</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">推理轮次</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{turns}轮</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 20px; border-radius: 10px; text-align: center;">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px;">耗时</div>
            <div style="color: white; font-weight: bold; font-size: 14px;">{elapsed}seconds</div>
        </div>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px; max-height: 400px; overflow-y: auto;">
        <div style="color: #333; font-weight: 700; font-size: 16px; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 2px solid #667eea;">
            📝 Final Answer
        </div>
        <div class="final-answer" style="color: #444; font-size: 15px; line-height: 1.8;">{simple_markdown_to_html(answer)}</div>
    </div>
</div>
"""
                    break

            if not final_html:
                if status == "running":
                    final_html = """
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 16px;">
    <div style="font-size: 36px; margin-bottom: 12px;">⏳</div>
    <div style="color: #666; font-size: 16px;">Research in Progress...</div>
</div>
"""
                else:
                    final_html = """
<div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 16px;">
    <div style="font-size: 36px; margin-bottom: 12px;">📋</div>
    <div style="color: #999; font-size: 14px;">WaitingResearch Complete...</div>
</div>
"""

            status_text = {"idle": "🟢 Idle", "running": "🔵 Running...", "completed": "✅ Completed", "error": "❌ Error"}.get(status, status)

            return process_html, final_html, status_text

        # 启动按钮点击事件
        def on_start(question, model_path, vllm_url, serper_key, jina_key,
                     summary_api_key, summary_api_base, summary_model_name,
                     proxy, max_steps):
            global global_state

            if not question.strip():
                return "<p style='color: red;'>⚠️ Please enter research question</p>", "<p></p>", "Error"

            if not model_path.strip():
                return "<p style='color: red;'>⚠️ Please enter model path</p>", "<p></p>", "Error"

            # Check Service
            vllm_url_check = vllm_url.rstrip("/")
            if not vllm_url_check.startswith("http"):
                vllm_url_check = f"http://{vllm_url_check}"
            is_ok, msg = check_vllm_server(vllm_url_check)
            if not is_ok:
                return f"<p style='color: red;'>⚠️ vLLM Service Not Ready: {msg}</p>", "<p></p>", "Error"

            # 启动线程
            thread = threading.Thread(target=start_research_thread, args=(
                question, model_path, vllm_url, serper_key, jina_key,
                summary_api_key, summary_api_base, summary_model_name,
                proxy, max_steps
            ))
            thread.start()

            return "<p style='color: #2196F3;'>🔄 推理已启动，请稍候...</p>", "<p style='color: #2196F3;'>🔄 Research in Progress...</p>", "启动中"

        start_btn.click(
            fn=on_start,
            inputs=[
                question_input, model_path, vllm_url, serper_key, jina_key,
                summary_api_key, summary_api_base, summary_model_name,
                proxy, max_steps
            ],
            outputs=[process_output, final_answer, status_display]
        )

        # 使用Timer组件实现定时刷新
        timer = gr.Timer(value=2)  # 每2seconds触发一次
        timer.tick(
            fn=get_status,
            inputs=[],
            outputs=[process_output, final_answer, status_display]
        )

        # 页面加载时清空状态并显示初始界面
        def on_page_load():
            global global_state
            # 清空之前的结果
            global_state["events"] = []
            global_state["status"] = "idle"
            global_state["is_running"] = False
            global_state["current_solver"] = None

            # 返回初始状态的HTML
            process_html = """
<div style="text-align: center; padding: 60px; color: #999;">
    <div style="font-size: 64px; margin-bottom: 20px;">🔍</div>
    <div style="font-size: 20px; font-weight: 600; color: #666;">DR-Venus</div>
    <div style="margin-top: 12px; font-size: 14px;">输入Research Question，开始深度研究之旅</div>
</div>
"""
            final_html = """
<div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 16px;">
    <div style="font-size: 36px; margin-bottom: 12px;">📋</div>
    <div style="color: #999; font-size: 14px;">WaitingResearch Complete...</div>
</div>
"""
            return process_html, final_html, "🟢 Idle"

        demo.load(
            fn=on_page_load,
            inputs=[],
            outputs=[process_output, final_answer, status_display]
        )

        # Stop按钮
        def on_stop():
            global global_state
            global_state["is_running"] = False
            return "<p style='color: orange;'>⏹️ 已发送Stop信号</p>", "<p></p>", "已Stop"

        stop_btn.click(
            fn=on_stop,
            inputs=[],
            outputs=[process_output, final_answer, status_display]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR-Venus Web Demo")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    demo = gradio_app()

    print("=" * 50)
    print("Starting DR-Venus Web Demo...")
    print("=" * 50)

    app, local_url, share_url = demo.launch(
        server_port=args.port,
        server_name="0.0.0.0",
        share=False,
        debug=args.debug,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="purple")
    )

    print("=" * 50)
    print(f"Local URL:  {local_url}")
    print(f"Share URL: {share_url}")
    print("=" * 50)