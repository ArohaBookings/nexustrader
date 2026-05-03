import { NextResponse, type NextRequest } from "next/server";
import { requireOwner } from "@/lib/api-auth";
import { getRisk } from "@/lib/repository";

export async function GET(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  return NextResponse.json(await getRisk());
}

