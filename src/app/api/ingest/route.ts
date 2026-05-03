import { NextResponse, type NextRequest } from "next/server";
import { requireSharedSecret } from "@/lib/api-auth";
import { ingestTelemetry } from "@/lib/repository";
import { ingestSchema } from "@/lib/validation";

export async function POST(request: NextRequest) {
  const auth = requireSharedSecret(request, "x-ingest-key", process.env.INGEST_API_KEY);
  if (auth) return auth;
  const parsed = ingestSchema.safeParse(await request.json());
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid_payload", issues: parsed.error.issues }, { status: 400 });
  }
  const result = await ingestTelemetry(parsed.data);
  return NextResponse.json({ ok: true, result });
}

