import { beforeEach, describe, expect, it, vi } from "vitest";
import { handleTelegramCommand } from "@/lib/commands";
import { resetMemoryStoreForTests } from "@/lib/test-store";

vi.mock("@/lib/bridge", () => ({
  executeBridgeControl: vi.fn(async (action: string) => ({ ok: true, action })),
}));

describe("Telegram command policy", () => {
  beforeEach(() => {
    delete process.env.DATABASE_URL;
    resetMemoryStoreForTests();
  });

  it("blocks trade placement and aggression requests", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "buy BTC now and increase leverage" });
    expect(reply).toContain("Blocked");
  });

  it("requires confirmation for resume", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "/resume" });
    expect(reply).toContain("Confirmation required");
    expect(reply).toContain("/confirm");
  });

  it("answers readonly status", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "/status" });
    expect(reply).toContain("Nexus Status");
    expect(reply).toContain("Equity");
  });
});

