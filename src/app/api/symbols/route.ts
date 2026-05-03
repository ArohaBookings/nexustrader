import { NextResponse, type NextRequest } from "next/server";
import { requireOwner } from "@/lib/api-auth";
import { getSymbols } from "@/lib/repository";

export async function GET(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  return NextResponse.json({ symbols: await getSymbols() });
}

