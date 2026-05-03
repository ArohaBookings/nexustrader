import { NextResponse, type NextRequest } from "next/server";
import { clearOwnerSessionCookie } from "@/lib/auth";

export async function GET(request: NextRequest) {
  await clearOwnerSessionCookie();
  return NextResponse.redirect(new URL("/login", request.url));
}
