"""Offline tests for the agent tools (Tavily / Jina / Hacker News) via respx."""

from __future__ import annotations

import httpx
import respx

from gateway.tools import build_default_tools
from gateway.tools.hackernews import make as make_hn
from gateway.tools.jina import make as make_jina
from gateway.tools.tavily import make as make_tavily


@respx.mock
def test_web_search_formats_answer_and_results():
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Paris is the capital of France.",
                "results": [
                    {
                        "title": "Paris",
                        "url": "https://example.com/paris",
                        "content": "City in France.",
                    }
                ],
            },
        )
    )
    out = make_tavily("tvly-key").run({"query": "capital of France"})
    assert "Answer: Paris is the capital of France." in out
    assert "https://example.com/paris" in out


@respx.mock
def test_web_search_requires_query():
    out = make_tavily("tvly-key").run({})
    assert out.startswith("Error:")


@respx.mock
def test_web_search_http_error_becomes_message(monkeypatch):
    monkeypatch.setenv("GATEWAY_RETRY_MAX_ATTEMPTS", "1")  # no retry/sleep in tests
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(500, text="boom"))
    out = make_tavily("tvly-key").run({"query": "x"})
    assert out.startswith("Error: web_search failed")


@respx.mock
def test_read_url_returns_text_and_truncates():
    target = "https://example.com/article"
    respx.get(f"https://r.jina.ai/{target}").mock(
        return_value=httpx.Response(200, text="A" * 9000)
    )
    out = make_jina(api_key=None).run({"url": target})
    assert out.endswith("…[truncated]")
    assert len(out) <= 8100


def test_read_url_rejects_relative_url():
    out = make_jina().run({"url": "not-a-url"})
    assert out.startswith("Error:")


@respx.mock
def test_read_url_sends_bearer_when_key_present():
    target = "https://example.com/x"
    route = respx.get(f"https://r.jina.ai/{target}").mock(
        return_value=httpx.Response(200, text="ok")
    )
    make_jina(api_key="jina-key").run({"url": target})
    assert route.calls.last.request.headers["authorization"] == "Bearer jina-key"


@respx.mock
def test_hackernews_search_formats_hits():
    respx.get(url__startswith="https://hn.algolia.com/api/v1/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Show HN: thing",
                        "url": "https://x.com",
                        "points": 42,
                        "num_comments": 7,
                    },
                    {"title": "No URL story", "objectID": "123", "points": 1, "num_comments": 0},
                ]
            },
        )
    )
    out = make_hn().run({"query": "rust"})
    assert "Show HN: thing (42 pts, 7 comments)" in out
    assert "https://news.ycombinator.com/item?id=123" in out  # falls back to HN item URL


def test_registry_includes_tavily_only_with_key():
    with_key = build_default_tools({"TAVILY_API_KEY": "k"})
    assert set(with_key) == {"web_search", "read_url", "hackernews_search"}

    without_key = build_default_tools({})
    assert "web_search" not in without_key
    assert set(without_key) == {"read_url", "hackernews_search"}
