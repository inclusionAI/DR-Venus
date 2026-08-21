'''
License: This code is adapted from Tongyi DeepResearch:
https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/tool_search.py
'''

import json
from typing import List, Union
from qwen_agent.tools.base import BaseTool, register_tool

from typing import Dict, List, Optional, Union

import http.client
import json
from dotenv import load_dotenv

import os
load_dotenv()

from tool_server.deadline import ToolDeadline, ToolTimeoutError, positive_timeout

SERPER_KEY=os.environ.get('SERPER_KEY_ID')
SERPER_TIMEOUT_SECONDS = positive_timeout(
    os.environ.get(
        "SERPER_TIMEOUT_SECONDS",
        os.environ.get("TOOL_TIMEOUT_SECONDS", os.environ.get("TOOL_TIMEOUT", 150)),
    ),
    name="SERPER_TIMEOUT_SECONDS",
)
#print("SERPER_KEY",SERPER_KEY)


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    def google_search_with_serp(
        self, query: str, *, deadline: ToolDeadline | None = None
    ):
        deadline = deadline or ToolDeadline.start(SERPER_TIMEOUT_SECONDS)

        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
            
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
            }
        
        
        last_error: Exception | None = None
        for i in range(5):
            conn = http.client.HTTPSConnection(
                "google.serper.dev", timeout=deadline.remaining()
            )
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                data = res.read()
                if res.status < 200 or res.status >= 300:
                    raise RuntimeError(f"Serper returned HTTP {res.status}")
                results = json.loads(data.decode("utf-8"))
                break
            except Exception as e:
                last_error = e
                print(e)
                if i == 4:
                    if isinstance(e, TimeoutError) or "timeout" in type(e).__name__.casefold():
                        raise ToolTimeoutError(
                            f"search timed out after {deadline.timeout_seconds:g} seconds"
                        ) from e
                    raise RuntimeError(f"Serper search failed after 5 attempts: {e}") from e
            finally:
                conn.close()

        if last_error is not None and "results" not in locals():
            raise RuntimeError(f"Serper search failed: {last_error}")

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    
    def search_with_serp(
        self, query: str, *, deadline: ToolDeadline | None = None
    ):
        result = self.google_search_with_serp(query, deadline=deadline)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        timeout = positive_timeout(
            kwargs.get("timeout", SERPER_TIMEOUT_SECONDS), name="search timeout"
        )
        deadline = ToolDeadline.start(timeout)
        try:
            query = params["query"]
            #print(query)
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # Single-query path.
            response = self.search_with_serp(query, deadline=deadline)
        else:
            # Batched multi-query path.
            if not isinstance(query, list):
                return "[Search] Invalid request format: 'query' must be a string or array"
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q, deadline=deadline))
            response = "\n=======\n".join(responses)
            print("response is :",response)

        # Prevent a provider response that arrives after expiry from being
        # treated (and recorded by the outer cache) as a successful result.
        deadline.remaining()
        return response
