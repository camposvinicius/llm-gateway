import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Gateway — Chat",
  description: "Multi-provider chat with per-token cost metering and fallback routing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
