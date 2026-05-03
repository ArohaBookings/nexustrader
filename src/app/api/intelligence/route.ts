import { NextResponse, type NextRequest } from "next/server";
import { requireOwner } from "@/lib/api-auth";
import { getOverview } from "@/lib/repository";

export async function GET(request: NextRequest) {
  const auth = requireOwner(request);
  if (auth) return auth;
  const overview = await getOverview();
  return NextResponse.json({ intelligence: overview.intelligence });
}
