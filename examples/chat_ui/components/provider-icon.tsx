import { siAnthropic, siGooglegemini, siOpenai } from "simple-icons/icons";
import { cn } from "@/lib/utils";

export type ProviderName = "bedrock" | "openai" | "gemini";

export type ModelOption = {
  id: "opus4.8" | "gemini3.1pro" | "gpt5.5";
  label: string;
  provider: ProviderName;
};

export const modelCatalog: ModelOption[] = [
  { id: "opus4.8", label: "Claude Opus 4.8", provider: "bedrock" },
  { id: "gemini3.1pro", label: "Gemini 3.1 Pro", provider: "gemini" },
  { id: "gpt5.5", label: "GPT-5.5", provider: "openai" },
];

export type ModelId = ModelOption["id"];

export function getModelById(id: string): ModelOption {
  return modelCatalog.find((m) => m.id === id) ?? modelCatalog[0];
}

type Meta = {
  label: string;
  short: string;
  icon: { path: string };
  color: string;
  bg: string;
  border: string;
};

export const providerMeta: Record<ProviderName, Meta> = {
  bedrock: { label: "Bedrock / Claude", short: "Claude", icon: siAnthropic, color: "text-[#D97757]", bg: "bg-[#D97757]/10", border: "border-[#D97757]/25" },
  openai: { label: "OpenAI", short: "OpenAI", icon: siOpenai, color: "text-[#10A37F]", bg: "bg-[#10A37F]/10", border: "border-[#10A37F]/25" },
  gemini: { label: "Gemini", short: "Gemini", icon: siGooglegemini, color: "text-[#8AB4F8]", bg: "bg-[#8AB4F8]/10", border: "border-[#8AB4F8]/25" },
};

export function getProviderMeta(provider: ProviderName | string): Meta {
  return providerMeta[(provider as ProviderName) in providerMeta ? (provider as ProviderName) : "bedrock"];
}

export function ProviderIcon({ provider, className }: { provider: ProviderName | string; className?: string }) {
  const meta = getProviderMeta(provider);
  return (
    <svg role="img" viewBox="0 0 24 24" className={cn("h-4 w-4", meta.color, className)} fill="currentColor" aria-label={meta.label}>
      <path d={meta.icon.path} />
    </svg>
  );
}
