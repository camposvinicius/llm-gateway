"""web_search tool backed by the Tavily Search API."""

from __future__ import annotations

import httpx

from .base import Tool, request_json

_ENDPOINT = "https://api.tavily.com/search"

_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {
            "type": "integer",
            "description": "How many results to return (1-10).",
        },
    },
    "required": ["query"],
}


def make(api_key: str) -> Tool:
    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        max_results = max(1, min(10, int(args.get("max_results") or 5)))
        try:
            data = request_json(
                "POST",
                _ENDPOINT,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
        except httpx.HTTPError as exc:
            return f"Error: web_search failed ({exc})."

        lines: list[str] = []
        if data.get("answer"):
            lines.append(f"Answer: {data['answer']}")
            lines.append("")
        for result in data.get("results", [])[:max_results]:
            snippet = (result.get("content") or "").strip().replace("\n", " ")[:300]
            title = result.get("title", "")
            url = result.get("url", "")
            lines.append(f"- {title}\n  {url}\n  {snippet}")
        return "\n".join(lines) if lines else "No results found."

    return Tool(
        name="web_search",
        description=(
            "Search the web for current information. Use this for recent events, "
            "news, prices, or anything that may have changed since training."
        ),
        input_schema=_SCHEMA,
        run=run,
    )
