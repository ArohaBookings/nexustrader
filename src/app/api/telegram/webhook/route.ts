import { NextResponse, type NextRequest } from "next/server";
import { handleTelegramCommand } from "@/lib/commands";
import { recordTelegramMessage } from "@/lib/repository";
import { extractTelegramMessage, sendTelegramMessage, telegramSecretValid } from "@/lib/telegram";
import { telegramUpdateSchema } from "@/lib/validation";

export async function POST(request: NextRequest) {
  if (!telegramSecretValid(request.headers.get("x-telegram-bot-api-secret-token"))) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const parsed = telegramUpdateSchema.safeParse(await request.json());
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid_payload", issues: parsed.error.issues }, { status: 400 });
  }
  const message = extractTelegramMessage(parsed.data);
  if (!message) return NextResponse.json({ ok: true, ignored: true });

  await recordTelegramMessage({
    chatId: message.chatId,
    telegramUserId: message.telegramUserId,
    username: message.username,
    direction: "in",
    text: message.text,
    payload: message.payload,
  });

  const reply = await handleTelegramCommand({
    chatId: message.chatId,
    telegramUserId: message.telegramUserId,
    text: message.text,
  });
  await recordTelegramMessage({
    chatId: message.chatId,
    telegramUserId: message.telegramUserId,
    username: message.username,
    direction: "out",
    text: reply,
    payload: { source: "telegram_webhook" },
  });
  const sent = await sendTelegramMessage(message.chatId, reply);
  return NextResponse.json({ ok: true, sent });
}

