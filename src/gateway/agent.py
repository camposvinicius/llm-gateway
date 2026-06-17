"""A small, gateway-native agent loop.

The gateway already normalizes tool calling across providers (see
``providers/base.py``) and meters every model call. This module turns those two
facts into a working research agent: call the model with tools, run any tool it
asks for, feed the results back, repeat until it answers or hits ``max_steps``.

Every model call goes through :meth:`Router.complete`, so each one is priced and
written to the ledger exactly like a plain ``/v1/chat`` request — the agent is a
loop *on top of* the gateway, not a bypass around its metering.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .messages import assistant_tool_use_turn, tool_result_turn
from .router import RoutedResponse, Router
from .tools import Tool

DEFAULT_MAX_STEPS = 6

# The agent has live tools, so it should answer from what they return — not hedge
# about a training cutoff. Injected only when the caller didn't supply a system turn.
AGENT_SYSTEM = (
    "You are a research assistant with live tools: web_search, read_url, and "
    "hackernews_search. Use them to answer with current information, and treat what "
    "they return as authoritative, up-to-date sources. Do NOT add disclaimers that "
    "your training data may be outdated or that you cannot verify recent events — you "
    "have live tools, so use them and report what they show. Cite a source link for "
    "each factual claim. If the tools genuinely don't return something, say so plainly. "
    "Write the final answer in clean, skimmable Markdown: a one-line summary first, then "
    "short sections or bullets, and a table when comparing things. "
    "Be efficient with tools: a few targeted searches beat many broad ones, and stop "
    "searching as soon as you have enough to answer."
)


@dataclass(frozen=True)
class AgentStep:
    """One executed tool call and its result, in order."""

    tool: str
    arguments: dict
    result: str


@dataclass(frozen=True)
class AgentResult:
    text: str
    steps: tuple[AgentStep, ...]
    input_tokens: int
    output_tokens: int
    cost_micro_usd: int
    model: str
    provider: str
    providers_tried: tuple[str, ...]
    request_ids: tuple[str, ...] = field(default_factory=tuple)
    stopped_at_max_steps: bool = False


def run_agent(
    router: Router,
    messages: list[dict],
    tools: dict[str, Tool],
    *,
    provider_chain: list[str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_model_response: Callable[[RoutedResponse], None] | None = None,
) -> AgentResult:
    convo: list[dict] = list(messages)
    if not any(m.get("role") == "system" for m in convo):
        convo = [{"role": "system", "content": AGENT_SYSTEM}, *convo]
    tool_defs = [tool.definition() for tool in tools.values()]

    steps: list[AgentStep] = []
    request_ids: list[str] = []
    total_input = total_output = total_cost = 0
    cache: dict[str, str] = {}  # reuse identical tool results to break re-fetch loops

    for _ in range(max(1, max_steps)):
        routed = router.complete(convo, provider_chain=provider_chain, tools=tool_defs)
        completion = routed.completion
        total_input += completion.input_tokens
        total_output += completion.output_tokens
        total_cost += routed.cost_micro_usd
        request_ids.append(routed.request_id)
        if on_model_response is not None:
            on_model_response(routed)

        if completion.stop_reason != "tool_use" or not completion.tool_calls:
            return _result(completion.text, steps, total_input, total_output, total_cost,
                           routed, request_ids, stopped_at_max_steps=False)

        # Record the model's tool-call turn, then run the tools and feed results back.
        convo.append(assistant_tool_use_turn(completion.text, completion.tool_calls))
        calls = list(completion.tool_calls)

        # Models (esp. GPT-5.x) emit many tool calls at once — run the uncached
        # ones concurrently so a turn costs the slowest call, not the sum.
        pending = [c for c in calls if _tool_key(c) not in cache]
        if pending:
            with ThreadPoolExecutor(max_workers=min(8, len(pending))) as pool:
                outputs = pool.map(lambda c: _run_tool(c, tools), pending)
                for call, output in zip(pending, outputs, strict=True):
                    cache[_tool_key(call)] = output

        results = []
        for call in calls:
            output = cache[_tool_key(call)]
            steps.append(AgentStep(tool=call.name, arguments=call.input, result=output))
            results.append({"tool_use_id": call.id, "name": call.name, "content": output})
        convo.append(tool_result_turn(results))

    # Ran out of steps while still calling tools. Force a final answer: drop the
    # tools and add an explicit instruction so the model writes a response now
    # instead of trying to call more tools (which left some models returning empty).
    convo.append(
        {
            "role": "user",
            "content": (
                "You've reached the research limit. Based on everything gathered above, "
                "write your final answer now with citations. Do not call any more tools."
            ),
        }
    )
    final = router.complete(convo, provider_chain=provider_chain, tools=None)
    total_input += final.completion.input_tokens
    total_output += final.completion.output_tokens
    total_cost += final.cost_micro_usd
    request_ids.append(final.request_id)
    if on_model_response is not None:
        on_model_response(final)

    return _result(final.completion.text or "(no answer produced)", steps,
                   total_input, total_output, total_cost, final, request_ids,
                   stopped_at_max_steps=True)


def _tool_key(call) -> str:
    return f"{call.name}:{json.dumps(call.input, sort_keys=True)}"


def _run_tool(call, tools: dict[str, Tool]) -> str:
    tool = tools.get(call.name)
    if tool is None:
        return f"Error: unknown tool {call.name!r}."
    try:
        return tool.run(call.input)
    except Exception as exc:  # a tool bug must not kill the request
        return f"Error running {call.name}: {exc}"


def _result(text, steps, total_input, total_output, total_cost, routed, request_ids,
            *, stopped_at_max_steps) -> AgentResult:
    return AgentResult(
        text=text,
        steps=tuple(steps),
        input_tokens=total_input,
        output_tokens=total_output,
        cost_micro_usd=total_cost,
        model=routed.completion.model,
        provider=routed.provider,
        providers_tried=routed.providers_tried,
        request_ids=tuple(request_ids),
        stopped_at_max_steps=stopped_at_max_steps,
    )
