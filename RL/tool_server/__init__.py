"""Open-source tool server for DeepResearcher / IGPO.

This package intentionally does NOT re-export anything at the top level.
Import specific symbols from their submodules directly, e.g.::

    from tool_server.execute_tools import custom_call_tool
    from tool_server.tool_prompt import SYSTEM_PROMPT, SUMMARY_PROMPT, EXTRACTOR_PROMPT

The no-reexport policy keeps ``import tool_server`` side-effect free: reading
the lightweight prompt strings in ``tool_prompt`` does not force loading the
heavy ``qwen_agent`` / ``tiktoken`` / ``openai`` stack pulled in by
``tool_search`` and ``tool_visit``.
"""
