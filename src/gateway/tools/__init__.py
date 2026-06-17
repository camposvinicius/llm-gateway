"""Tool registry for the research agent.

``build_default_tools`` returns the tools whose prerequisites are satisfied by
the environment: web_search needs a Tavily key, read_url uses an optional Jina
key, hackernews_search needs nothing. Tools without their key are simply not
registered, so the agent degrades gracefully instead of failing at call time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from . import hackernews, jina, tavily
from .base import Tool

__all__ = ["Tool", "build_default_tools"]


def build_default_tools(env: Mapping[str, str] | None = None) -> dict[str, Tool]:
    env = env if env is not None else os.environ
    tools: dict[str, Tool] = {}

    tavily_key = env.get("TAVILY_API_KEY")
    if tavily_key:
        tools["web_search"] = tavily.make(tavily_key)

    tools["read_url"] = jina.make(env.get("JINA_API_KEY"))
    tools["hackernews_search"] = hackernews.make()
    return tools
