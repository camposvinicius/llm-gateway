"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Check,
  ChevronDown,
  Clock3,
  DollarSign,
  Loader2,
  MessageSquarePlus,
  Route,
  SendHorizonal,
  Sparkles,
  TerminalSquare,
  Trash2,
  Wrench,
} from "lucide-react";
import { ModelId, ProviderIcon, getModelById, getProviderMeta, modelCatalog } from "@/components/provider-icon";
import { compactNumber, cn, usdFromMicroUsd } from "@/lib/utils";

type Role = "user" | "assistant";

type ToolStep = {
  tool: string;
  arguments: Record<string, unknown>;
  result: string;
};

type GatewayResponse = {
  id: string;
  provider: string;
  model: string;
  text: string;
  usage: { input_tokens: number; output_tokens: number };
  cost: { micro_usd: number };
  routing: { providers_tried: string[] };
  steps?: ToolStep[];
  tools_available?: string[];
  stopped_at_max_steps?: boolean;
};

type Message = {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  meta?: GatewayResponse & { latency_ms: number; display_model: string };
};

type Conversation = {
  id: string;
  title: string;
  modelId: ModelId;
  messages: Message[];
  updatedAt: number;
};

const storageKey = "llm-gateway-chat-ui:v3";
const oldStorageKey = "llm-gateway-chat-ui:v1";

const starterPrompts = [
  "Compare the three selected frontier models for a startup support bot.",
  "Explain why an LLM gateway needs pricing, fallback, and a ledger.",
  "Write a production incident update for a degraded upstream LLM provider.",
];

const agentStarterPrompts = [
  "What's the latest news about Anthropic's Claude models?",
  "What are people on Hacker News saying about local LLMs this week?",
  "Find and summarize the top 3 articles on the state of AI agents in 2026.",
];

function newConversation(modelId: ModelId = "opus4.8"): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New gateway chat",
    modelId,
    messages: [],
    updatedAt: Date.now(),
  };
}

function titleFrom(content: string) {
  const cleaned = content.replace(/\s+/g, " ").trim();
  return cleaned.length > 44 ? `${cleaned.slice(0, 44)}…` : cleaned || "New gateway chat";
}

function providerChain(modelId: ModelId) {
  return [getModelById(modelId).provider];
}

function toolArgSummary(args: Record<string, unknown>): string {
  const value = args?.query ?? args?.url ?? Object.values(args ?? {})[0];
  return typeof value === "string" ? value : JSON.stringify(args ?? {});
}

export default function Page() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [agentMode, setAgentMode] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    window.localStorage.removeItem(oldStorageKey);
    const saved = window.localStorage.getItem(storageKey);
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as { conversations: Conversation[]; activeId: string };
        const valid = parsed.conversations?.filter((conversation) => modelCatalog.some((model) => model.id === conversation.modelId));
        if (valid?.length) {
          setConversations(valid);
          setActiveId(valid.some((conversation) => conversation.id === parsed.activeId) ? parsed.activeId : valid[0].id);
          return;
        }
      } catch {
        window.localStorage.removeItem(storageKey);
      }
    }
    const initial = newConversation("opus4.8");
    setConversations([initial]);
    setActiveId(initial.id);
  }, []);

  useEffect(() => {
    if (!conversations.length) return;
    window.localStorage.setItem(storageKey, JSON.stringify({ conversations, activeId }));
  }, [conversations, activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, activeId, isSending]);

  const active = conversations.find((conversation) => conversation.id === activeId) ?? conversations[0];
  const activeModel = getModelById(active?.modelId ?? "opus4.8");
  const selectedMeta = getProviderMeta(activeModel.provider);
  const lastAssistant = [...(active?.messages ?? [])].reverse().find((message) => message.role === "assistant");

  const totals = useMemo(() => {
    const all = conversations.flatMap((conversation) => conversation.messages);
    return all.reduce(
      (acc, message) => {
        if (message.meta) {
          acc.requests += 1;
          acc.microUsd += message.meta.cost.micro_usd;
          acc.inputTokens += message.meta.usage.input_tokens;
          acc.outputTokens += message.meta.usage.output_tokens;
        }
        return acc;
      },
      { requests: 0, microUsd: 0, inputTokens: 0, outputTokens: 0 },
    );
  }, [conversations]);

  function updateActive(updater: (conversation: Conversation) => Conversation) {
    setConversations((current) => current.map((conversation) => (conversation.id === activeId ? updater(conversation) : conversation)));
  }

  function createConversation(modelId: ModelId = active?.modelId ?? "opus4.8") {
    const conversation = newConversation(modelId);
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setInput("");
    setError(null);
  }

  function clearConversation() {
    if (!active) return;
    updateActive((conversation) => ({ ...conversation, messages: [], title: "New gateway chat", updatedAt: Date.now() }));
    setError(null);
  }

  function deleteConversation(id: string) {
    setConversations((current) => {
      const remaining = current.filter((conversation) => conversation.id !== id);
      if (!remaining.length) {
        const fresh = newConversation("opus4.8");
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(remaining[0].id);
      return remaining;
    });
    setError(null);
  }

  function deleteMessage(messageId: string) {
    updateActive((conversation) => ({
      ...conversation,
      messages: conversation.messages.filter((message) => message.id !== messageId),
      updatedAt: Date.now(),
    }));
  }

  async function sendMessage(contentOverride?: string) {
    if (!active || isSending) return;
    const content = (contentOverride ?? input).trim();
    if (!content) return;

    setInput("");
    setError(null);
    setIsSending(true);

    const modelAtSend = getModelById(active.modelId);
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content, createdAt: Date.now() };
    const messagesForApi = [...active.messages, userMessage].map((message) => ({ role: message.role, content: message.content }));

    updateActive((conversation) => ({
      ...conversation,
      title: conversation.messages.length ? conversation.title : titleFrom(content),
      messages: [...conversation.messages, userMessage],
      updatedAt: Date.now(),
    }));

    const started = performance.now();
    try {
      const chain = providerChain(modelAtSend.id);
      const response = await fetch(agentMode ? "/api/agent" : "/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(
          agentMode
            ? { messages: messagesForApi, provider_chain: chain, max_steps: 4 }
            : { messages: messagesForApi, provider_chain: chain },
        ),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? `Gateway returned HTTP ${response.status}`);

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: payload.text ?? "",
        createdAt: Date.now(),
        meta: { ...(payload as GatewayResponse), latency_ms: Math.round(performance.now() - started), display_model: modelAtSend.label },
      };

      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === activeId
            ? { ...conversation, messages: [...conversation.messages, assistantMessage], updatedAt: Date.now() }
            : conversation,
        ),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsSending(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  if (!active) return null;

  return (
    <main className="grid h-screen grid-cols-[344px_minmax(0,1fr)_340px] gap-3 p-3 text-foreground">
      <aside className="glass flex min-h-0 flex-col rounded-3xl">
        <div className="border-b border-white/10 p-4">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-accent/10 text-accent shadow-glow">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold tracking-wide">llm-gateway</p>
              <p className="text-xs text-muted">local multi-provider console</p>
            </div>
          </div>
          <button
            onClick={() => createConversation()}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl border border-accent/25 bg-accent/10 px-3 py-2.5 text-sm font-semibold text-accent transition hover:bg-accent/15"
          >
            <MessageSquarePlus className="h-4 w-4" /> New chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted">Conversations</p>
          <div className="space-y-2">
            {conversations.map((conversation) => {
              const model = getModelById(conversation.modelId);
              const isActive = conversation.id === active.id;
              return (
                <div
                  key={conversation.id}
                  className={cn(
                    "group flex w-full items-center gap-2 rounded-2xl border p-3 text-left transition",
                    isActive ? "border-accent/30 bg-accent/10" : "border-white/5 bg-white/[0.03] hover:bg-white/[0.06]",
                  )}
                >
                  <button onClick={() => setActiveId(conversation.id)} className="flex min-w-0 flex-1 flex-col text-left">
                    <span className="flex items-center gap-2">
                      <ProviderIcon provider={model.provider} />
                      <span className="truncate text-sm font-medium">{conversation.title}</span>
                    </span>
                    <span className="mt-1 truncate text-xs text-muted">{model.label} · {conversation.messages.length} messages</span>
                  </button>
                  <button
                    onClick={() => deleteConversation(conversation.id)}
                    title="Delete conversation"
                    className="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-muted/80 transition hover:bg-red-400/15 hover:text-red-200"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div className="border-t border-white/10 p-4">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <MetricMini label="Requests" value={String(totals.requests)} />
            <MetricMini label="Cost" value={usdFromMicroUsd(totals.microUsd)} />
            <MetricMini label="In tok" value={compactNumber(totals.inputTokens)} />
            <MetricMini label="Out tok" value={compactNumber(totals.outputTokens)} />
          </div>
        </div>
      </aside>

      <section className="glass relative z-20 flex min-h-0 min-w-0 flex-col rounded-3xl">
        <header className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold">{active.title}</h1>
            <p className="text-xs text-muted">Gateway-backed chat with metering, fallback, and ledger writes.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setAgentMode((value) => !value)}
              title={agentMode ? "Agent mode on — the gateway can call web tools" : "Turn on agent mode (web search, URL reader, Hacker News)"}
              className={cn(
                "flex h-12 items-center gap-2 rounded-2xl border px-3.5 text-sm font-medium transition",
                agentMode
                  ? "border-accent/40 bg-accent/15 text-accent shadow-glow"
                  : "border-white/10 bg-panel2 text-muted hover:text-foreground hover:bg-white/10",
              )}
            >
              <Wrench className="h-4 w-4" /> Agent
            </button>
            <ModelDropdown
              activeModelId={active.modelId}
              open={modelMenuOpen}
              setOpen={setModelMenuOpen}
              onSelect={(modelId) => updateActive((conversation) => ({ ...conversation, modelId, updatedAt: Date.now() }))}
            />
            <button
              onClick={clearConversation}
              className="grid h-12 w-12 place-items-center rounded-2xl border border-white/10 bg-panel2 text-muted transition hover:bg-white/10 hover:text-foreground"
              title="Clear messages in this chat"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6">
          {active.messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center text-center">
              <div className={cn("mb-5 grid h-16 w-16 place-items-center rounded-3xl border", selectedMeta.bg, selectedMeta.border)}>
                <ProviderIcon provider={activeModel.provider} className="h-8 w-8" />
              </div>
              <h2 className="text-2xl font-semibold">
                {agentMode ? "Ask the research agent." : "Ask the gateway anything."}
              </h2>
              <p className="mt-2 max-w-xl text-sm leading-6 text-muted">
                {agentMode
                  ? "Ask something that needs fresh information. The gateway routes to the selected model, lets it call web tools, and shows every tool call, the route, tokens, latency, and cost."
                  : "Choose one of the three models, send a prompt, and watch the gateway report the real provider, model, tokens, route, latency, and cost. Toggle Agent to let it use web tools."}
              </p>
              <div className="mt-7 grid w-full max-w-3xl gap-3 md:grid-cols-3">
                {(agentMode ? agentStarterPrompts : starterPrompts).map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => void sendMessage(prompt)}
                    className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-left text-sm leading-5 transition hover:border-accent/30 hover:bg-accent/10"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-4xl space-y-5">
              {active.messages.map((message) => (
                <ChatBubble key={message.id} message={message} onDelete={() => deleteMessage(message.id)} />
              ))}
              {isSending && (
                <div className="flex items-center gap-3 text-sm text-muted">
                  <Loader2 className="h-4 w-4 animate-spin text-accent" />
                  {agentMode ? "Agent is researching with tools…" : "Gateway is routing the request…"}
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {error && <div className="mx-5 mb-3 rounded-2xl border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-100">{error}</div>}

        <form onSubmit={onSubmit} className="border-t border-white/10 p-4">
          <div className="flex items-end gap-3 rounded-3xl border border-white/10 bg-panel2 p-2 shadow-2xl">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder="Message the gateway…"
              rows={1}
              className="max-h-40 min-h-11 flex-1 resize-none bg-transparent px-3 py-3 text-sm outline-none placeholder:text-muted"
            />
            <button type="submit" disabled={isSending || !input.trim()} className="grid h-11 w-11 place-items-center rounded-2xl bg-accent text-black transition hover:brightness-110 disabled:opacity-40">
              {isSending ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizonal className="h-4 w-4" />}
            </button>
          </div>
        </form>
      </section>

      <aside className="glass flex min-h-0 min-w-0 flex-col rounded-3xl">
        <div className="border-b border-white/10 p-4">
          <p className="text-sm font-semibold">Gateway trace</p>
          <p className="text-xs text-muted">Last assistant response</p>
        </div>
        {lastAssistant?.meta ? <TracePanel meta={lastAssistant.meta} /> : <EmptyTrace />}
      </aside>
    </main>
  );
}

function ModelDropdown({ activeModelId, open, setOpen, onSelect }: { activeModelId: ModelId; open: boolean; setOpen: (value: boolean) => void; onSelect: (modelId: ModelId) => void }) {
  const selected = getModelById(activeModelId);
  return (
    <div className="relative z-50 w-[260px]">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex h-12 w-full items-center justify-between gap-3 rounded-2xl border border-white/10 bg-[#1f1f27] px-4 text-left text-sm text-foreground shadow-xl transition hover:border-white/20 hover:bg-[#252530]"
      >
        <span className="flex min-w-0 items-center gap-3">
          <ProviderIcon provider={selected.provider} className="h-5 w-5 shrink-0" />
          <span className="truncate">{selected.label}</span>
        </span>
        <ChevronDown className={cn("h-4 w-4 shrink-0 text-muted transition", open && "rotate-180")} />
      </button>
      {open && (
        <div className="absolute right-0 z-[100] mt-2 w-[300px] overflow-hidden rounded-2xl border border-white/10 bg-[#202026] py-2 shadow-[0_18px_80px_rgba(0,0,0,0.55)]">
          {modelCatalog.map((model) => {
            const active = model.id === activeModelId;
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => {
                  onSelect(model.id);
                  setOpen(false);
                }}
                className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-[15px] text-[#f3f4f6] transition hover:bg-white/[0.07]"
              >
                <span className="flex min-w-0 items-center gap-3">
                  <ProviderIcon provider={model.provider} className="h-5 w-5 shrink-0" />
                  <span className="truncate">{model.label}</span>
                </span>
                {active && <Check className="h-4 w-4 text-accent" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MetricMini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-white/[0.035] p-3">
      <p className="text-[10px] uppercase tracking-[0.14em] text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm text-foreground">{value}</p>
    </div>
  );
}

function ChatBubble({ message, onDelete }: { message: Message; onDelete: () => void }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("group flex gap-3", isUser && "justify-end")}>
      {!isUser && (
        <div className="mt-1 grid h-9 w-9 shrink-0 place-items-center rounded-2xl border border-accent/20 bg-accent/10 text-accent">
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div className={cn("relative rounded-3xl border px-4 py-3", isUser ? "max-w-[78%] border-accent/20 bg-accent/15" : "max-w-[88%] border-white/10 bg-white/[0.045]")}>
        <button
          onClick={onDelete}
          title="Delete message"
          className="absolute -right-2 -top-2 grid h-7 w-7 place-items-center rounded-full border border-white/10 bg-[#202026] text-muted opacity-0 shadow-xl transition hover:bg-red-400/15 hover:text-red-200 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        {!isUser && message.meta?.steps && message.meta.steps.length > 0 && (
          <div className="mb-3 rounded-2xl border border-accent/20 bg-accent/[0.06] p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-accent">
              <Wrench className="h-3.5 w-3.5" />
              Agent used {message.meta.steps.length} tool call{message.meta.steps.length > 1 ? "s" : ""}
            </div>
            <ol className="space-y-1.5">
              {message.meta.steps.map((step, index) => (
                <li key={index} className="flex gap-2 text-[12px] leading-5">
                  <span className="text-muted/60">{index + 1}.</span>
                  <span className="min-w-0">
                    <span className="font-mono font-medium text-foreground">{step.tool}</span>
                    <span className="text-muted"> · {toolArgSummary(step.arguments)}</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}
        {isUser ? (
          <div className="whitespace-pre-wrap text-sm leading-6">{message.content}</div>
        ) : (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        )}
        {message.meta && (
          <div className="mt-3 flex flex-wrap gap-2 border-t border-white/10 pt-3 text-[11px] text-muted">
            <span className="inline-flex items-center gap-1"><ProviderIcon provider={message.meta.provider} /> {message.meta.display_model}</span>
            <span>{message.meta.provider}</span>
            <span>{message.meta.model}</span>
            <span>{usdFromMicroUsd(message.meta.cost.micro_usd)}</span>
            <span>{message.meta.latency_ms}ms</span>
            {message.meta.steps && message.meta.steps.length > 0 && (
              <span className="inline-flex items-center gap-1 text-accent">
                <Wrench className="h-3 w-3" /> {message.meta.steps.length} tool call{message.meta.steps.length > 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TracePanel({ meta }: { meta: GatewayResponse & { latency_ms: number; display_model: string } }) {
  const providerMeta = getProviderMeta(meta.provider);
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="rounded-3xl border border-white/10 bg-white/[0.035] p-4">
        <div className="flex items-center gap-3">
          <div className={cn("grid h-11 w-11 place-items-center rounded-2xl border", providerMeta.bg, providerMeta.border)}>
            <ProviderIcon provider={meta.provider} className="h-6 w-6" />
          </div>
          <div>
            <p className="font-semibold">{meta.display_model}</p>
            <p className="max-w-[220px] truncate text-xs text-muted">real: {meta.model}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3">
        <TraceMetric icon={<DollarSign className="h-4 w-4" />} label="Cost" value={usdFromMicroUsd(meta.cost.micro_usd)} />
        <TraceMetric icon={<Clock3 className="h-4 w-4" />} label="Latency" value={`${meta.latency_ms}ms`} />
        <TraceMetric icon={<Activity className="h-4 w-4" />} label="Input" value={`${meta.usage.input_tokens} tok`} />
        <TraceMetric icon={<TerminalSquare className="h-4 w-4" />} label="Output" value={`${meta.usage.output_tokens} tok`} />
      </div>

      <div className="mt-4 rounded-3xl border border-white/10 bg-white/[0.035] p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Route className="h-4 w-4 text-accent" /> Providers tried</div>
        <div className="space-y-2">
          {meta.routing.providers_tried.map((provider, index) => (
            <div key={`${provider}-${index}`} className="flex items-center justify-between rounded-2xl border border-white/5 bg-panel2 px-3 py-2 text-sm">
              <span className="flex items-center gap-2"><ProviderIcon provider={provider} /> {provider}</span>
              <span className={cn("text-xs", provider === meta.provider ? "text-accent" : "text-muted")}>{provider === meta.provider ? "served" : "failed"}</span>
            </div>
          ))}
        </div>
      </div>

      {meta.steps && meta.steps.length > 0 && (
        <div className="mt-4 rounded-3xl border border-white/10 bg-white/[0.035] p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold">
            <Wrench className="h-4 w-4 text-accent" /> Tool calls
            {meta.stopped_at_max_steps && (
              <span className="ml-auto text-[10px] uppercase tracking-wide text-amber-300/80">max steps</span>
            )}
          </div>
          <div className="space-y-3">
            {meta.steps.map((step, index) => (
              <div key={index} className="rounded-2xl border border-white/5 bg-panel2 p-3">
                <p className="text-sm font-medium text-accent">{step.tool}</p>
                <pre className="mt-1 overflow-x-auto text-[11px] text-muted">{JSON.stringify(step.arguments)}</pre>
                <p className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-xs leading-5 text-muted">{step.result}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 rounded-3xl border border-white/10 bg-black/20 p-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted">Raw response</p>
        <pre className="max-h-80 overflow-auto text-xs leading-5 text-muted">{JSON.stringify(meta, null, 2)}</pre>
      </div>
    </div>
  );
}

function TraceMetric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.035] p-4">
      <div className="mb-2 text-accent">{icon}</div>
      <p className="text-[10px] uppercase tracking-[0.16em] text-muted">{label}</p>
      <p className="mt-1 font-mono text-sm">{value}</p>
    </div>
  );
}

function EmptyTrace() {
  return <div className="flex flex-1 items-center justify-center p-8 text-center text-sm leading-6 text-muted">Send a message to see provider, real model, route, tokens, latency, and cost.</div>;
}
