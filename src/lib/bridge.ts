const ACTION_TO_ENDPOINT = {
  pause_trading: "pause_trading",
  resume_trading: "resume_trading",
  kill_switch: "kill_switch",
  refresh_state: "refresh_state",
} as const;

export type BridgeAction = keyof typeof ACTION_TO_ENDPOINT;

function bridgeBaseUrl() {
  return (process.env.BRIDGE_API_BASE_URL ?? "").replace(/\/+$/, "");
}

function bridgeHeaders(extra?: HeadersInit) {
  const headers = new Headers(extra);
  const token = process.env.BRIDGE_TOKEN;
  if (token) headers.set("x-bridge-token", token);
  return headers;
}

async function bridgeLogin(baseUrl: string) {
  const password = process.env.BRIDGE_DASHBOARD_PASSWORD ?? process.env.APP_ADMIN_PASSWORD;
  if (!password) return "";
  const response = await fetch(`${baseUrl}/dashboard/login`, {
    method: "POST",
    body: new URLSearchParams({ password }),
    redirect: "manual",
  });
  if (!response.ok && response.status !== 303) {
    throw new Error(`Bridge dashboard login failed with HTTP ${response.status}`);
  }
  return response.headers.get("set-cookie") ?? "";
}

export async function executeBridgeControl(action: BridgeAction) {
  const baseUrl = bridgeBaseUrl();
  if (!baseUrl) {
    return {
      ok: false,
      action,
      reason: "bridge_not_configured",
    };
  }
  const endpoint = ACTION_TO_ENDPOINT[action];
  const cookie = await bridgeLogin(baseUrl);
  const headers = bridgeHeaders(cookie ? { cookie } : undefined);
  const response = await fetch(`${baseUrl}/dashboard/control/${endpoint}`, {
    method: "POST",
    headers,
  });
  const body = await safeJson(response);
  return {
    ok: response.ok,
    action,
    status: response.status,
    bridge: body,
  };
}

export async function fetchBridgeJson(path: string) {
  const baseUrl = bridgeBaseUrl();
  if (!baseUrl) return null;
  const response = await fetch(`${baseUrl}${path.startsWith("/") ? path : `/${path}`}`, {
    headers: bridgeHeaders(),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Bridge ${path} failed with HTTP ${response.status}`);
  }
  return safeJson(response);
}

async function safeJson(response: Response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { raw: text };
  }
}

