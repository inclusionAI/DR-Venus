'''
License: This code is adapted from Tongyi DeepResearch:
https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/react_agent.py
'''

import json
import json5
import os
import re
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from datetime import datetime

from tool_server.tool_search import *
from tool_server.tool_visit import *


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
  #要从tool_call dict中提取出tool_name:str  tool_args: dict
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]
    tool_args["params"] = tool_args
    if tool_name == "search":
        raw_result = Search().call(tool_args, **kwargs)
        result = parse_search_results(raw_result)
        return result
    elif tool_name == "visit":
        raw_result = Visit().call(tool_args, **kwargs)
        result = raw_result
        return result
    else:
        return f"Error: Tool {tool_name} not found"

#print(custom_call_tool({"name":"search","arguments":{"query":['Fire Station 301 DCA ARFF metro station']}}))
