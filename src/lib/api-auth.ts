import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { requestHasOwnerSession } from "@/lib/auth";

export function requireOwner(request: NextRequest) {
  if (!requestHasOwnerSession(request)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  return null;
}

export function bearerOrHeader(request: NextRequest, headerName: string) {
  const direct = request.headers.get(headerName);
  if (direct) return direct;
  const auth = request.headers.get("authorization") ?? "";
  if (auth.toLowerCase().startsWith("bearer ")) return auth.slice(7).trim();
  return "";
}

export function requireSharedSecret(request: NextRequest, headerName: string, expected: string | undefined) {
  const supplied = bearerOrHeader(request, headerName);
  if (!expected || supplied !== expected) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  return null;
}

