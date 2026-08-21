'''
License: This code is adapted from Tongyi DeepResearch:
https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/tool_visit.py
'''

import json
from dotenv import load_dotenv
import os

from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI

try:
    from .deadline import (
        ToolDeadline,
        ToolTimeoutError,
        is_timeout_error,
        positive_timeout,
    )
    from .tool_prompt import EXTRACTOR_PROMPT
except ImportError:  # Support ``python Inference/run_demo.py``.
    from tool_server.deadline import (
        ToolDeadline,
        ToolTimeoutError,
        is_timeout_error,
        positive_timeout,
    )
    from tool_server.tool_prompt import EXTRACTOR_PROMPT

from urllib.parse import urlparse, unquote
import time 

import tiktoken

load_dotenv()

VISIT_SERVER_TIMEOUT = positive_timeout(
    os.getenv("VISIT_SERVER_TIMEOUT", os.getenv("TOOL_TIMEOUT_SECONDS", 150)),
    name="VISIT_SERVER_TIMEOUT",
)
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
PROXY = os.getenv("PROXY", "")


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        timeout = positive_timeout(
            kwargs.get("timeout", VISIT_SERVER_TIMEOUT), name="visit timeout"
        )
        deadline = ToolDeadline.start(timeout)
        try:
            url = params["url"]
            goal = params["goal"]
            #print(f"url:{url},goal:{goal}")
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            response = self.readpage_jina(url, goal, deadline=deadline)
        else:
            response = []
            if not isinstance(url, list):
                return "[Visit] Invalid request format: 'url' must be a string or array"
            for u in url: 
                # All URLs share one deadline so a large batch cannot multiply
                # the configured per-tool wall-clock budget.
                deadline.remaining()
                cur_response = self.readpage_jina(u, goal, deadline=deadline)
                response.append(cur_response)
            response = "\n=======\n".join(response)

        deadline.remaining()
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
        
    def call_server(self, msgs, max_retries=2, *, deadline: ToolDeadline):
        max_retries = max(1, int(max_retries))
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        for attempt in range(max_retries):
            client = OpenAI(
                api_key=api_key,
                base_url=url_llm,
                timeout=deadline.remaining(),
                max_retries=0,
            )
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                if is_timeout_error(e):
                    raise ToolTimeoutError(
                        f"visit summarizer timed out after {deadline.timeout_seconds:g} seconds"
                    ) from e
                if attempt == (max_retries - 1):
                    return ""
                continue
            finally:
                client.close()
        return ""


    def jina_readpage(self, url: str, *, deadline: ToolDeadline) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {JINA_API_KEYS}",
            }
            proxies = {
                "http": f"{PROXY}",
                "https": f"{PROXY}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    proxies=proxies,
                    headers=headers,
                    timeout=deadline.remaining(),
                )
                #print("response is :",response)
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print("response.text is :",response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if is_timeout_error(e):
                    raise ToolTimeoutError(
                        f"Jina read timed out after {deadline.timeout_seconds:g} seconds"
                    ) from e
                deadline.sleep(0.5)
                print("[visit] Failed to read page.")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str, *, deadline: ToolDeadline) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            deadline.remaining()
            content = self.jina_readpage(url, deadline=deadline)
            service = "jina"     
            print(service)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage_jina(
        self, url: str, goal: str, *, deadline: ToolDeadline
    ) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
   
        summary_page_func = self.call_server
        max_retries = max(1, int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1)))

        content = self.html_readpage_jina(url, deadline=deadline)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(
                messages, max_retries=max_retries, deadline=deadline
            )
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(
                    messages, max_retries=max_retries, deadline=deadline
                )
                summary_retries -= 1

            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(
                        messages, max_retries=max_retries, deadline=deadline
                    )
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            return useful_information

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information
