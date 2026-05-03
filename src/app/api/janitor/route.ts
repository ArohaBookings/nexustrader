import { NextResponse, type NextRequest } from "next/server";
import { requireSharedSecret } from "@/lib/api-auth";
import { runJanitor } from "@/lib/repository";

export async function POST(request: NextRequest) {
  const auth = requireSharedSecret(request, "x-janitor-secret", process.env.JANITOR_SECRET);
  if (auth) return auth;
  return NextResponse.json({ ok: true, result: await runJanitor() });
}

export async function GET(request: NextRequest) {
  return POST(request);
}

