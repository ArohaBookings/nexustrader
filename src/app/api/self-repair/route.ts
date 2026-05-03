import { NextResponse, type NextRequest } from "next/server";
import { requireSharedSecret } from "@/lib/api-auth";
import { runSelfRepairAudit } from "@/lib/self-repair";

function repairSecret() {
  return process.env.SELF_REPAIR_SECRET ?? process.env.CRON_SECRET ?? process.env.JANITOR_SECRET;
}

export async function POST(request: NextRequest) {
  const auth = requireSharedSecret(request, "x-self-repair-secret", repairSecret());
  if (auth) return auth;
  return NextResponse.json(await runSelfRepairAudit("api_self_repair"));
}

export async function GET(request: NextRequest) {
  return POST(request);
}
