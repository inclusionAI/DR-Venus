#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os
import random
import re
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import filelock
import json5
import openai
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

#from tools_server.http_client import MessageClient
from tool_server.execute_tools import custom_call_tool

# Suppress verbose library logging (only keep WARNING+)
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
    "visit": {"url", "goal"},}

ALLOWED_TOOLS = set(ALLOWED_ARGS.keys())


def _filter_args(tool_name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        return {}
    allowed = ALLOWED_ARGS.get(tool_name, set())
    return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ResearchProblemSolver:
    def __init__(
        self,
        question: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        openai_client: openai.Client,
        # tool_client: MessageClient,
        model_name: str,
        max_steps: int,
        time_limit: int = 9000,
        verbose: bool = False,
        max_retries: int = 5,
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
        # self.tool_client = tool_client
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max(max_retries, 1)  # at least 1 attempt

        self.start_time = time.time()
        self.turns = 0
        self.parse_error_count = 0
        self.tool_call_count = 0
        self.dedup_count = 0

        # Tool call dedup history: key -> True
        self._tool_call_history: dict[str, bool] = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"[Solver] {msg}")

    # ---- tool call dedup ----

    @staticmethod
    def _tool_call_key(tool_call: dict) -> str:
        """Generate a hashable key from tool name + arguments for dedup."""
        return json.dumps(
            {"name": tool_call["name"], "arguments": tool_call["arguments"]},
            sort_keys=True, ensure_ascii=False,
        )

    # ---- repetition detection ----

    @staticmethod
    def _is_repetitive(content: str, tail_len: int = 50, threshold: int = 5) -> bool:
        """Check if the last `tail_len` chars appear more than `threshold` times."""
        if not content or len(content) < tail_len:
            return False
        tail = content[-tail_len:]
        return content.count(tail) > threshold

    # ---- token counting ----

    def count_tokens(self) -> int:
        rendered = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return len(
            self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )

    # ---- single LLM call (no retry) ----

    def _call_llm_once(self) -> str | None:
        """Single API call. Returns stripped content or None on failure."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                stop=["<tool_response>", "\n<tool_response>"],
                temperature=1.0,
                max_tokens=15000,
                top_p=0.95,
                presence_penalty=1.1,
                extra_body={
                    "top_k": 20,
                }, 
            )
            content = response.choices[0].message.content
            return content.strip() if content and content.strip() else None
        except Exception as e:
            self.log(f"API error: {e}")
            return None

    # ---- response parsing (regex + status code) ----

    def parse_llm_response(self, content: str, step: int) -> tuple[int, str, object]:
        """
        Returns (status, thinking_or_error, payload):
          0 = final answer   -> payload is answer text
          1 = tool call      -> payload is tool_call dict
          2 = parse error    -> payload is ""
        """
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

        # Final answer
        if answer_match and not tool_match:
            return 0, thinking, answer_match.group(1).strip()

        # Tool call
        if tool_match and not answer_match:
            raw_tool_json = tool_match.group(1).strip()

            # Other tools: JSON parse
            try:
                tool_call = json5.loads(raw_tool_json)

                if not isinstance(tool_call, dict):
                    return 2, f"Tool call should be dict, got {type(tool_call)}", ""
                if "name" not in tool_call or "arguments" not in tool_call:
                    return 2, f"Tool call missing fields, raw={raw_tool_json}", ""

                if tool_call["name"] not in ALLOWED_TOOLS:
                    return 2, f"Unknown tool '{tool_call['name']}'. Available tools: {', '.join(sorted(ALLOWED_TOOLS))}", ""

                tool_call["arguments"] = _filter_args(
                    tool_call["name"], tool_call["arguments"]
                )
                return 1, thinking, tool_call

            except Exception as e:
                return 2, f"Tool call parse error: {repr(e)}, raw={raw_tool_json}", ""

        return 2, "Ambiguous or incomplete output", ""

    # ---- tool execution via tongyi tool ----

    def execute_tool(self, tool_call: dict) -> str:
      #传入tool_call，返回tool结果content
        try:
            #results = self.tool_client.submit_tasks(query)
            results = custom_call_tool(tool_call)
            if results and len(results) > 0:
                return results
            return "ERROR: empty tool result"
        except Exception as e:
            return f"ERROR: tool execution failed: {e}"

    # ---- extract answer from message history ----

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

    # ---- structured result builder ----

    def _make_result(self, termination: str, prediction: str | None = None) -> dict:
        if prediction is None:
            prediction = self.extract_final_answer() or "No answer found"
        print(prediction)
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

    # ---- sanitize LLM output ----

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

    # ---- unified generate with retry ----

    def _generate(self, step: int) -> tuple[str | None, int, str, object]:
        """
        Call LLM and validate output quality with unified retry for:
          1. API failure / empty response
          2. Repetitive output
          3. Format parse error (status=2)
          4. Duplicate tool call (dedup)

        Returns (content, status, thinking, payload).
        content=None means LLM is completely unavailable.
        """
        # Track last results for fallback when retries are exhausted
        last_content = None
        last_status = -1
        last_thinking = ""
        last_payload = None

        for attempt in range(self.max_retries):
            is_last = (attempt == self.max_retries - 1)
            backoff = min(2 ** attempt + random.uniform(0, 1), 30)

            # 1. API call
            content = self._call_llm_once()
            if not content:
                self.log(f"[_generate] API failure (attempt {attempt + 1}/{self.max_retries})")
                if not is_last:
                    time.sleep(backoff)
                    continue
                # All retries exhausted with API failure
                return None, -1, "", None

            # 2. Repetition check
            if self._is_repetitive(content):
                self.log(f"[_generate] Repeat detected (attempt {attempt + 1}/{self.max_retries})")
                last_content = content
                if not is_last:
                    time.sleep(backoff)
                    continue
                # Last attempt: use repetitive content anyway
                self.log("[_generate] Repeat detected after all retries, using anyway")

            # 3. Sanitize + parse
            content = self._sanitize_content(content)
            status, thinking, payload = self.parse_llm_response(content, step)

            # 4. Parse error → retry
            if status == 2:
                self.parse_error_count += 1
                self.log(f"[_generate] Parse error (attempt {attempt + 1}/{self.max_retries}): {thinking}")
                last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                if not is_last:
                    time.sleep(backoff)
                    continue
                # Last attempt: return parse error for solve() to handle
                return content, status, thinking, payload

            # 5. Duplicate tool call → retry
            if status == 1:
                key = self._tool_call_key(payload)
                if key in self._tool_call_history:
                    self.dedup_count += 1
                    self.log(f"[_generate] Duplicate tool call (attempt {attempt + 1}/{self.max_retries}): {payload['name']}")
                    last_content, last_status, last_thinking, last_payload = content, status, thinking, payload
                    if not is_last:
                        time.sleep(backoff)
                        continue
                    # Last attempt: execute the duplicate call anyway
                    return content, status, thinking, payload

            # All checks passed
            return content, status, thinking, payload

        # Should not reach here, but return last result as fallback
        if last_content is not None:
            return last_content, last_status, last_thinking, last_payload
        return None, -1, "", None

    # ---- force answer (token limit / max steps) ----

    def _force_answer(self, termination_ok: str, termination_fail: str) -> dict:
        """Force LLM to produce a final answer (used at token/step limits)."""
        content, status, thinking, payload = self._generate(step=-1)
        if content is None:
            return self._make_result(termination_fail)

        self.messages.append({"role": "assistant", "content": content})
        self.turns += 1

        if status == 0:
            return self._make_result(termination_ok, payload)

        # status != 0: LLM didn't produce <answer>, fall back to raw content.
        return self._make_result(termination_fail)

    # ---- main loop ----

    def solve(self) -> dict:
        for step in range(self.max_steps):
            # Timeout check
            if time.time() - self.start_time > self.time_limit:
                return self._make_result("timeout")

            # Unified generate with retry
            content, status, thinking, payload = self._generate(step)
            if content is None:
                return self._make_result("llm_error")

            self.messages.append({"role": "assistant", "content": content})

            if status == 0:  # final answer
                self.turns += 1
                return self._make_result("answered", payload)

            if status == 1:  # tool call
                self.turns += 1
                self.tool_call_count += 1
                self._tool_call_history[self._tool_call_key(payload)] = True
                result = self.execute_tool(payload)
                self.messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>",
                })

            if status == 2:  # parse error (retries exhausted)
                self.turns += 1
                self.log(f"Parse error at step {step}: {thinking}")
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"<tool_response>\nERROR!\n"
                        "Please call a tool with the correct format, or if you are ready to answer, provide the entire final answer in `<answer></answer>` tags.\n</tool_response>"
                    ),
                })

            # Token limit check
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

        # max_steps exhausted, force a final answer
        self.messages[-1] = {
            "role": "user",
            "content": (
                "<tool_response>\nYou've reached the maximum number of tool calls. "
                "Based on the information and knowledge you have gathered, please provide the entire final answer in `<answer></answer>` tags now.\n</tool_response>"
            ),
        }
        return self._force_answer("answered_max_steps", "max_steps")


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_tp(args):
    cmd = [
        "vllm", "serve", args.model_path,
        "--served-model-name", "model",
        "--gpu-memory-utilization", str(args.gpu_util),
        "-tp", str(args.tp_size),
        "--max-num-seqs", str(args.batch_size),
        "--max-model-len", str(args.max_model_len),
        "--enforce-eager",
        #"--disable-cuda-graph",
        # "--disable-log-stats",
        #"--disable-log-requests",0
        "--port", str(args.base_port),
    ]
    proc = subprocess.Popen(cmd)
    return [proc], [args.base_port]


def start_vllm_multi(args):
    procs = []
    ports = []
    for i in range(args.num_gpus):
        port = args.base_port + i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            "vllm", "serve", args.model_path,
            "--served-model-name", "model",
            "--gpu-memory-utilization", str(args.gpu_util),
            "--max-num-seqs", str(args.batch_size),
            "--max-model-len", str(args.max_model_len),
            # "--disable-log-stats",
            # "--disable-log-requests",
            "--port", str(port),
        ]
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        ports.append(port)
    return procs, ports


def start_vllm(args) -> tuple[list, list[int]]:
    """Start vLLM servers. Returns (process_list, port_list)."""
    if args.deploy_mode == "tp":
        expected_ports = [args.base_port]
    else:
        expected_ports = [args.base_port + i for i in range(args.num_gpus)]

    # Check if servers are already running
    all_running = True
    for port in expected_ports:
        try:
            requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
        except Exception:
            all_running = False
            break

    if all_running:
        print(f"Detected existing vLLM servers on ports {expected_ports}")
        return [], expected_ports

    print(f"Starting vLLM servers (mode={args.deploy_mode})...")
    if args.deploy_mode == "tp":
        procs, ports = start_vllm_tp(args)
    else:
        procs, ports = start_vllm_multi(args)

    wait_for_servers(ports)
    return procs, ports


def wait_for_servers(ports: list[int], timeout: int = 6000):
    start = time.time()
    ready = set()
    while time.time() - start < timeout:
        for port in ports:
            if port in ready:
                continue
            try:
                requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                print(f"vLLM server on port {port} is ready.")
                ready.add(port)
            except Exception:
                pass
        if len(ready) == len(ports):
            print("All vLLM servers are ready.")
            return
        time.sleep(5)
    raise RuntimeError(
        f"vLLM servers failed to start within {timeout}s. "
        f"Ready: {ready}, Expected: {set(ports)}"
    )


def stop_vllm(procs: list):
    if not procs:
        return
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("All vLLM processes stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_openai_client(port: int) -> openai.Client:
    return openai.Client(
        api_key="123",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


# def build_tool_client() -> MessageClient:
#     return MessageClient(isconsumer=True)


def append_jsonl_safely(output_path: str, lock_path: str, data: dict):
    with filelock.FileLock(lock_path):
        with open(output_path, "a+", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Batch Inference Pipeline (v2)")

    # Deployment
    parser.add_argument("--deploy_mode", type=str, default="tp",
                        choices=["tp", "multi"],
                        help="tp: single server with tensor parallelism; "
                             "multi: one server per GPU")
    parser.add_argument("--tp_size", type=int, default=2,
                        help="Tensor parallel size (tp mode only)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs (multi mode only)")
    parser.add_argument("--base_port", type=int, default=6000,
                        help="Base port for vLLM servers")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_util", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=261000)

    # Data
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    # Inference
    parser.add_argument("--batch_size", type=int, default=32,
                        help="vLLM max-num-seqs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Thread pool concurrency")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per problem")
    parser.add_argument("--solver_max_len", type=int, default=240000,
                        help="Solver token limit")
    parser.add_argument("--time_limit", type=int, default=9000,
                        help="Per-problem time limit in seconds (default 150min)")
    parser.add_argument("--verbose", action="store_true")

    # Robustness (unified)
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retries for API errors, repetition, parse errors, "
                             "and duplicate tool calls (0 = no retry)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    vllm_procs, ports = start_vllm(args)

    try:
        print(f"Loading tokenizer from: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        output_path = args.output_file
        lock_path = output_path + ".lock"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


        questions_to_process = []
        q = args.question
        questions_to_process.append(q)

        print(f"Pending samples: {len(questions_to_process)}")
        print(f"Workers: {args.num_workers}, Ports: {ports}")

        stats_lock = Lock()
        stats = {"success": 0, "failed": 0}
        progress = tqdm(total=len(questions_to_process), desc="Processing")

        def solve_problem(question: str, worker_index: int):
            port = ports[worker_index % len(ports)]
            solver = None
            try:
                local_openai_client = build_openai_client(port)
                # local_tool_client = build_tool_client()

                solver = ResearchProblemSolver(
                    question=question,
                    tokenizer=tokenizer,
                    max_len=args.solver_max_len,
                    max_steps=args.max_steps,
                    openai_client=local_openai_client,
                    # tool_client=local_tool_client,
                    model_name="model",
                    time_limit=args.time_limit,
                    verbose=args.verbose,
                    max_retries=args.max_retries,
                )

                result = solver.solve()

                record = {
                    "question": question,
                    "messages": solver.messages,
                    "prediction": result["prediction"],
                    "termination": result["termination"],
                    "turns": result["turns"],
                    "parse_error_count": result["parse_error_count"],
                    "tool_call_count": result["tool_call_count"],
                    "dedup_count": result["dedup_count"],
                    "final_token_count": result["final_token_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
                append_jsonl_safely(output_path, lock_path, record)

                with stats_lock:
                    if result["termination"].startswith("answered"):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

            except Exception as e:
                tb = traceback.format_exc()
                record = {
                    "question": question,
                    "messages": solver.messages if solver else None,
                    "prediction": None,
                    "termination": "exception",
                    "turns": 0,
                    "parse_error_count": 0,
                    "tool_call_count": 0,
                    "dedup_count": 0,
                    "final_token_count": 0,
                    "elapsed_seconds": 0,
                    "error": repr(e),
                    "traceback": tb,
                }
                append_jsonl_safely(output_path, lock_path, record)
                print(f"\n[ERROR] {question[:80]}...: {repr(e)}")
                if args.verbose:
                    print(tb)
                with stats_lock:
                    stats["failed"] += 1

            finally:
                progress.update(1)

        if questions_to_process:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(solve_problem, q, i)
                    for i, q in enumerate(questions_to_process)
                ]
                for future in as_completed(futures):
                    future.result()

        progress.close()

        print(f"\nDone. Success: {stats['success']}, Failed: {stats['failed']}")
        print(f"Output: {output_path}")

    finally:
        stop_vllm(vllm_procs)
