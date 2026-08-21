'''
License: This code is adapted from Tongyi DeepResearch:
https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/react_agent.py
'''

import json
import json5
import os
from pathlib import Path
import re
import sys
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from datetime import datetime

try:
    from .tool_search import Search
    from .tool_visit import Visit
except ImportError:  # Support ``python Inference/run_demo.py``.
    from tool_server.tool_search import Search
    from tool_server.tool_visit import Visit

try:
    from drvenus_repro.tool_cache import execute_with_tool_cache
except ModuleNotFoundError as exc:
    if exc.name not in {"drvenus_repro", "drvenus_repro.tool_cache"}:
        raise
    # Inference is normally launched after ``cd Inference``; make the shared
    # repository-level reproducibility package importable in that workflow.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from drvenus_repro.tool_cache import execute_with_tool_cache


def parse_search_results(text):
    #  找到所有 query 位置
    query_pattern = r"A Google search for '([^']+)'"
    query_matches = list(re.finditer(query_pattern, text))

    all_results = []

    for i, match in enumerate(query_matches):
        query = match.group(1)

        # 当前 query 的内容范围
        start = match.end()
        end = query_matches[i + 1].start() if i + 1 < len(query_matches) else len(text)
        chunk = text[start:end]

        # 提取该 query 对应的搜索结果
        result_pattern = r'\d+\.\s+\[(.*?)\]\((.*?)\)\s*\n+(.*?)(?=\n\d+\.\s+\[|\Z)'
        matches = re.findall(result_pattern, chunk, re.DOTALL)

        results = []
        for title, url, summary in matches:
            results.append({
                'title': title.strip(),
                'url': url.strip(),
                'snippet': summary.strip()
            })

        all_results.append({
            'query': query,
            'web_page_info_list': results
        })

    return json.dumps(all_results, ensure_ascii=False)

def custom_call_tool(tool_call: dict, **kwargs):
    # 要从 tool_call dict 中提取出 tool_name: str、tool_args: dict。
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]
    if tool_name not in {"search", "visit"}:
        return f"Error: Tool {tool_name} not found"

    def call_online():
        if tool_name == "search":
            return parse_search_results(Search().call(tool_args, **kwargs))
        return Visit().call(tool_args, **kwargs)

    provider = "serper" if tool_name == "search" else "jina+openai-compatible-summary"
    return execute_with_tool_cache(tool_name, tool_args, call_online, provider=provider)

#print(custom_call_tool({"name":"search","arguments":{"query":['Fire Station 301 DCA ARFF metro station']}}))
