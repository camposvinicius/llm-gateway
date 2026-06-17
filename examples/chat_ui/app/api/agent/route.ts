import { NextRequest, NextResponse } from "next/server";

// Server-side proxy to the gateway's research agent. Same pattern as /api/chat:
// keeps the browser same-origin and the gateway URL server-only.
const GATEWAY_URL = process.env.GATEWAY_URL || "http://127.0.0.1:8080";

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "invalid JSON body" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${GATEWAY_URL}/v1/agent`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return NextResponse.json(
      { detail: `gateway unreachable at ${GATEWAY_URL}: ${String(err)}` },
      { status: 502 },
    );
  }

  const text = await upstream.text();
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = { detail: text || upstream.statusText };
  }

  return NextResponse.json(payload, { status: upstream.status });
}
