# web_research_agent

A tiny CLI over the gateway's research agent (`POST /v1/agent`). The agent loop,
the tools (web search, URL reader, Hacker News), and the per-call metering all
live in the gateway — this script just sends a question and prints the tool
trace plus the final answer.

## Run

Start the gateway with tool keys set (see the repo `.env.example` — `TAVILY_API_KEY`
enables `web_search`, `JINA_API_KEY` is optional for `read_url`, Hacker News needs
no key), then:

```bash
# from the repo root, with the gateway running on :8080
python examples/web_research_agent/research.py "What's the latest news about Claude Opus?"

# pick a provider and bound the tool-use rounds
python examples/web_research_agent/research.py --provider gemini --max-steps 4 \
  "What are people on Hacker News saying about local LLMs?"
```

Set `GATEWAY_URL` if the gateway is not on `http://127.0.0.1:8080`.

## What you'll see

```
Question: What's the latest news about Claude Opus?
Model: global.anthropic.claude-opus-4-8 (via bedrock)
Tools available: hackernews_search, read_url, web_search

Tool calls:
  1. web_search({'query': 'latest Claude Opus news'})
     → Answer: ...
  2. read_url({'url': 'https://...'})
     → ...

Answer:
<the synthesized answer>

420 in / 380 out tokens · $0.012345 · 2 tool call(s)
```
