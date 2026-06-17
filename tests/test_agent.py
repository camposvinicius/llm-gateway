"""Offline tests for the research-agent loop and the /v1/agent endpoint.

A scripted provider returns a fixed sequence of completions so the loop is
deterministic without any network or credentials.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gateway.agent import run_agent
from gateway.app import create_app
from gateway.ledger import Ledger
from gateway.metrics import GatewayMetrics
from gateway.pricing import PricingTable
from gateway.providers.base import Completion, ToolCall
from gateway.router import Router
from gateway.tools.base import Tool


class ScriptedProvider:
    name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[list, list | None]] = []

    def complete(self, messages, tools=None):
        self.calls.append((messages, tools))
        return self._responses.pop(0)


@pytest.fixture()
def pricing(tmp_path):
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {"models": {"scripted-1": {"input_per_million": "100", "output_per_million": "200"}}}
        )
    )
    return PricingTable.from_file(path)


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "agent.db")
    yield led
    led.close()


def _completion(text="", tool_calls=(), stop_reason="end_turn", model="scripted-1"):
    return Completion(
        text=text,
        model=model,
        input_tokens=1,
        output_tokens=1,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
    )


def _tool(name="search", result="RESULT"):
    return Tool(
        name=name,
        description="desc",
        input_schema={"type": "object", "properties": {}},
        run=lambda args: f"{result}:{args}",
    )


def test_agent_runs_tool_then_answers(pricing, ledger):
    provider = ScriptedProvider(
        [
            _completion(tool_calls=(ToolCall(id="t1", name="search", input={"q": "x"}),),
                        stop_reason="tool_use"),
            _completion(text="final answer"),
        ]
    )
    router = Router([provider], pricing, ledger)

    result = run_agent(router, [{"role": "user", "content": "hi"}], {"search": _tool()})

    assert result.text == "final answer"
    assert len(result.steps) == 1
    assert result.steps[0].tool == "search"
    assert result.steps[0].arguments == {"q": "x"}
    assert result.steps[0].result == "RESULT:{'q': 'x'}"
    assert len(result.request_ids) == 2  # both model calls metered
    assert result.cost_micro_usd > 0
    assert len(ledger.entries()) == 2

    # The second model call must have received the tool_result turn.
    second_messages = provider.calls[1][0]
    assert any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
        for m in second_messages
    )


def test_agent_unknown_tool_is_reported(pricing, ledger):
    provider = ScriptedProvider(
        [
            _completion(tool_calls=(ToolCall(id="t1", name="missing", input={}),),
                        stop_reason="tool_use"),
            _completion(text="ok"),
        ]
    )
    router = Router([provider], pricing, ledger)

    result = run_agent(router, [{"role": "user", "content": "hi"}], {})

    assert result.steps[0].result.startswith("Error: unknown tool")
    assert result.text == "ok"


def test_agent_failing_tool_is_captured_not_raised(pricing, ledger):
    def boom(_args):
        raise RuntimeError("kaboom")

    tool = Tool(name="search", description="d", input_schema={}, run=boom)
    provider = ScriptedProvider(
        [
            _completion(tool_calls=(ToolCall(id="t1", name="search", input={}),),
                        stop_reason="tool_use"),
            _completion(text="recovered"),
        ]
    )
    router = Router([provider], pricing, ledger)

    result = run_agent(router, [{"role": "user", "content": "hi"}], {"search": tool})

    assert "kaboom" in result.steps[0].result
    assert result.text == "recovered"


def test_agent_max_steps_forces_final_synthesis(pricing, ledger):
    tool_call = (ToolCall(id="t1", name="search", input={}),)
    provider = ScriptedProvider(
        [
            _completion(tool_calls=tool_call, stop_reason="tool_use"),
            _completion(tool_calls=tool_call, stop_reason="tool_use"),
            _completion(text="synthesized"),  # final no-tools synthesis call
        ]
    )
    router = Router([provider], pricing, ledger)

    result = run_agent(
        router, [{"role": "user", "content": "hi"}], {"search": _tool()}, max_steps=2
    )

    assert result.stopped_at_max_steps is True
    assert result.text == "synthesized"
    assert len(result.steps) == 2
    # The synthesis call must omit tools so the model is forced to answer.
    assert provider.calls[-1][1] is None


def test_agent_endpoint_returns_steps_and_meters(pricing, ledger):
    provider = ScriptedProvider(
        [
            _completion(tool_calls=(ToolCall(id="t1", name="search", input={"q": "x"}),),
                        stop_reason="tool_use"),
            _completion(text="final"),
        ]
    )
    router = Router([provider], pricing, ledger)
    app = create_app(router=router, metrics=GatewayMetrics(), tools={"search": _tool()})
    client = TestClient(app)

    response = client.post("/v1/agent", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "final"
    assert body["steps"][0]["tool"] == "search"
    assert body["steps"][0]["arguments"] == {"q": "x"}
    assert body["tools_available"] == ["search"]
    assert body["usage"]["input_tokens"] == 2  # summed across both model calls
    assert len(ledger.entries()) == 2


def test_agent_endpoint_rejects_missing_messages(pricing, ledger):
    router = Router([ScriptedProvider([_completion(text="x")])], pricing, ledger)
    app = create_app(router=router, metrics=GatewayMetrics(), tools={})
    client = TestClient(app)

    response = client.post("/v1/agent", json={})
    assert response.status_code == 400
