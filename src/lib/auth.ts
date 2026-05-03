import crypto from "node:crypto";
import { cookies } from "next/headers";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

const COOKIE_NAME = "nexus_session";
const SESSION_TTL_SECONDS = 60 * 60 * 12;

type SessionPayload = {
  sub: "owner";
  iat: number;
  exp: number;
};

function base64url(input: string) {
  return Buffer.from(input).toString("base64url");
}

function unbase64url(input: string) {
  return Buffer.from(input, "base64url").toString("utf8");
}

function sessionSecret() {
  const secret = process.env.SESSION_SECRET;
  if (!secret || secret.length < 16) {
    throw new Error("SESSION_SECRET must be at least 16 characters");
  }
  return secret;
}

function sign(payload: string) {
  return crypto.createHmac("sha256", sessionSecret()).update(payload).digest("base64url");
}

export function createSessionValue(now = Math.floor(Date.now() / 1000)) {
  const payload: SessionPayload = {
    sub: "owner",
    iat: now,
    exp: now + SESSION_TTL_SECONDS,
  };
  const encoded = base64url(JSON.stringify(payload));
  return `${encoded}.${sign(encoded)}`;
}

export function verifySessionValue(value: string | undefined | null, now = Math.floor(Date.now() / 1000)) {
  if (!value) return false;
  const [encoded, signature] = value.split(".");
  if (!encoded || !signature) return false;
  const expected = sign(encoded);
  if (signature.length !== expected.length) return false;
  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) return false;
  try {
    const payload = JSON.parse(unbase64url(encoded)) as SessionPayload;
    return payload.sub === "owner" && Number(payload.exp) > now;
  } catch {
    return false;
  }
}

export async function hasOwnerSession() {
  const jar = await cookies();
  return verifySessionValue(jar.get(COOKIE_NAME)?.value);
}

export async function setOwnerSessionCookie() {
  const jar = await cookies();
  jar.set(COOKIE_NAME, createSessionValue(), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
  });
}

export async function clearOwnerSessionCookie() {
  const jar = await cookies();
  jar.delete(COOKIE_NAME);
}

export function requestHasOwnerSession(request: NextRequest) {
  return verifySessionValue(request.cookies.get(COOKIE_NAME)?.value);
}

export function unauthorized() {
  return NextResponse.json({ error: "unauthorized" }, { status: 401 });
}

export function timingSafePasswordMatches(input: string) {
  const expected = process.env.APP_ADMIN_PASSWORD;
  if (!expected || expected.length < 8) {
    return false;
  }
  const inputHash = crypto.createHash("sha256").update(input).digest();
  const expectedHash = crypto.createHash("sha256").update(expected).digest();
  return crypto.timingSafeEqual(inputHash, expectedHash);
}
