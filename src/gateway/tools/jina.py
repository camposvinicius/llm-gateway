"""read_url tool backed by the Jina Reader (r.jina.ai).

Fetches a URL and returns clean, LLM-friendly text/markdown. A Jina key is
optional (it raises rate limits); the tool works without one.
"""

from __future__ import annotations

import httpx

from .base import Tool, request_text

_READER_PREFIX = "https://r.jina.ai/"
# Keep reads small: the agent often reads several pages, and each result is
# re-sent on every subsequent step, so large dumps blow up context, cost, and
# latency (especially on reasoning models).
_MAX_CHARS = 3500

_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The absolute URL to read."},
    },
    "required": ["url"],
}


def make(api_key: str | None = None) -> Tool:
    def run(args: dict) -> str:
        url = (args.get("url") or "").strip()
        if not url:
            return "Error: 'url' is required."
        if not url.startswith(("http://", "https://")):
            return "Error: 'url' must be an absolute http(s) URL."
        headers = {"Accept": "text/plain"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            text = request_text("GET", f"{_READER_PREFIX}{url}", headers=headers)
        except httpx.HTTPError as exc:
            return f"Error: read_url failed ({exc})."
        text = text.strip()
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n\n…[truncated]"
        return text or "No readable content."

    return Tool(
        name="read_url",
        description=(
            "Fetch a web page and return its main content as clean text. "
            "Use after web_search to read a specific result in full."
        ),
        input_schema=_SCHEMA,
        run=run,
    )
