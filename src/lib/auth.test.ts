import { describe, expect, it } from "vitest";
import { createSessionValue, timingSafePasswordMatches, verifySessionValue } from "@/lib/auth";

describe("auth sessions", () => {
  it("signs and verifies owner sessions", () => {
    const value = createSessionValue(1000);
    expect(verifySessionValue(value, 1001)).toBe(true);
    expect(verifySessionValue(`${value}x`, 1001)).toBe(false);
    expect(verifySessionValue(value, 1000 + 60 * 60 * 13)).toBe(false);
  });

  it("checks passwords without length-sensitive comparison failures", () => {
    process.env.APP_ADMIN_PASSWORD = "test-password";
    expect(timingSafePasswordMatches("bad")).toBe(false);
    expect(timingSafePasswordMatches("test-password")).toBe(true);
  });
});

