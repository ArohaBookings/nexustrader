import type { TelegramUpdate } from "@/lib/validation";

export function telegramSecretValid(headerValue: string | null) {
  const expected = process.env.TELEGRAM_WEBHOOK_SECRET;
  return Boolean(expected && headerValue === expected);
}

export function extractTelegramMessage(update: TelegramUpdate) {
  const message = update.message;
  if (!message?.chat?.id || !message.text) return null;
  return {
    chatId: String(message.chat.id),
    text: message.text.trim(),
    telegramUserId: message.from?.id ? String(message.from.id) : undefined,
    username: message.from?.username,
    payload: update as unknown as Record<string, unknown>,
  };
}

export async function sendTelegramMessage(chatId: string, text: string) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  if (!token) {
    return { ok: false, reason: "telegram_not_configured" };
  }
  const response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      chat_id: chatId,
      text,
      parse_mode: "Markdown",
      disable_web_page_preview: true,
    }),
  });
  const body = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, body };
}

