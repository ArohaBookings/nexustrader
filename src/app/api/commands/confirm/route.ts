import { NextResponse, type NextRequest } from "next/server";
import { requireOwner } from "@/lib/api-auth";
import { executePendingCommand } from "@/lib/commands";

export async function POST(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  const body = (await request.json()) as { commandId?: string };
  if (!body.commandId) {
    return NextResponse.json({ error: "commandId_required" }, { status: 400 });
  }
  const result = await executePendingCommand(body.commandId);
  return NextResponse.json({ ok: true, result });
}

