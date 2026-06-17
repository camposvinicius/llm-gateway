#!/usr/bin/env python3
"""Tiny CLI for the gateway's research agent.

It is deliberately thin: the agent loop, the tools, and the metering all live in
the gateway (POST /v1/agent). This script just sends a question and pretty-prints
the tool trace and the final answer — proof that the agent is a server feature,
not client glue.

Usage:
    python research.py "What's the latest news about Claude Opus?"
    python research.py --provider gemini --max-steps 4 "Compare Rust vs Go in 2026"

Env:
    GATEWAY_URL   gateway base URL (default http://127.0.0.1:8080)
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gateway research agent.")
    parser.add_argument("question", nargs="+", help="The question to research.")
    parser.add_argument(
        "--provider",
        default="bedrock",
        help="Provider to route to (bedrock|openai|gemini). Default: bedrock.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Max tool-use rounds.")
    parser.add_argument(
        "--gateway",
        default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080"),
        help="Gateway base URL.",
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    payload = {
        "messages": [{"role": "user", "content": question}],
        "provider_chain": [args.provider],
        "max_steps": args.max_steps,
    }

    try:
        response = httpx.post(f"{args.gateway}/v1/agent", json=payload, timeout=180)
    except httpx.HTTPError as exc:
        print(f"error: could not reach gateway at {args.gateway}: {exc}", file=sys.stderr)
        return 1

    if response.status_code != 200:
        print(
            f"error: gateway returned HTTP {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        return 1

    data = response.json()

    print(f"\n\033[1mQuestion:\033[0m {question}")
    print(f"\033[1mModel:\033[0m {data.get('model')} (via {data.get('provider')})")
    available = ", ".join(data.get("tools_available", [])) or "none"
    print(f"\033[1mTools available:\033[0m {available}\n")

    steps = data.get("steps", [])
    if steps:
        print("\033[1mTool calls:\033[0m")
        for i, step in enumerate(steps, 1):
            result = (step.get("result") or "").strip().replace("\n", " ")
            print(f"  {i}. {step['tool']}({step['arguments']})")
            print(f"     → {result[:160]}{'…' if len(result) > 160 else ''}")
        print()

    print("\033[1mAnswer:\033[0m")
    print(data.get("text", ""))

    usage = data.get("usage", {})
    cost = data.get("cost", {}).get("micro_usd", 0)
    print(
        f"\n\033[2m{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens "
        f"· ${cost / 1_000_000:.6f} · {len(steps)} tool call(s)\033[0m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
