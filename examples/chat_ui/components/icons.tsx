// Provider glyphs — original geometric SVGs evoking each brand, not their
// official logos. currentColor so they adapt to theme.

export function BedrockIcon({ className = "" }: { className?: string }) {
  // Anthropic-style "burst" of radial lines.
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
      {Array.from({ length: 8 }).map((_, i) => {
        const a = (Math.PI * 2 * i) / 8;
        const x1 = 12 + Math.cos(a) * 4;
        const y1 = 12 + Math.sin(a) * 4;
        const x2 = 12 + Math.cos(a) * 9;
        const y2 = 12 + Math.sin(a) * 9;
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />;
      })}
    </svg>
  );
}

export function OpenAIIcon({ className = "" }: { className?: string }) {
  // Interlocked hexagonal knot, stylized.
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
      <path d="M12 3l7 4v8l-7 4-7-4V7z" />
      <path d="M12 7l3.5 2v4L12 15l-3.5-2V9z" />
    </svg>
  );
}

export function GeminiIcon({ className = "" }: { className?: string }) {
  // Four-point sparkle / star.
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor">
      <path d="M12 2c.6 5 2.8 7.4 8 8-5.2.6-7.4 3-8 8-.6-5-2.8-7.4-8-8 5.2-.6 7.4-3 8-8z" />
    </svg>
  );
}

export const PROVIDERS = [
  { id: "bedrock", label: "Bedrock", Icon: BedrockIcon, tint: "#d97757" },
  { id: "openai", label: "OpenAI", Icon: OpenAIIcon, tint: "#10a37f" },
  { id: "gemini", label: "Gemini", Icon: GeminiIcon, tint: "#4285f4" },
] as const;

export type ProviderId = (typeof PROVIDERS)[number]["id"];
