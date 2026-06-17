"""hackernews_search tool backed by the Algolia Hacker News API (no key)."""

from __future__ import annotations

import httpx

from .base import Tool, request_json

_ENDPOINT = "https://hn.algolia.com/api/v1/search"

_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search Hacker News for."},
    },
    "required": ["query"],
}


def make() -> Tool:
    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' is required."
        try:
            data = request_json("GET", _ENDPOINT, params={"query": query, "tags": "story"})
        except httpx.HTTPError as exc:
            return f"Error: hackernews_search failed ({exc})."

        lines: list[str] = []
        for hit in data.get("hits", [])[:5]:
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            lines.append(
                f"- {hit.get('title', '(no title)')} "
                f"({hit.get('points', 0)} pts, {hit.get('num_comments', 0)} comments)\n  {url}"
            )
        return "\n".join(lines) if lines else "No stories found."

    return Tool(
        name="hackernews_search",
        description=(
            "Search Hacker News stories by keyword. Good for developer sentiment, "
            "launches, and discussion around a technology or company."
        ),
        input_schema=_SCHEMA,
        run=run,
    )
