import { NextResponse, type NextRequest } from "next/server";
import { requireOwner } from "@/lib/api-auth";
import { getFundedOverview, saveFundedConfig } from "@/lib/repository";
import { fundedConfigSchema } from "@/lib/validation";

export async function GET(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  return NextResponse.json(await getFundedOverview());
}

export async function POST(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  const parsed = fundedConfigSchema.safeParse(await request.json());
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid_payload", issues: parsed.error.issues }, { status: 400 });
  }
  await saveFundedConfig(parsed.data);
  return NextResponse.json(await getFundedOverview());
}
