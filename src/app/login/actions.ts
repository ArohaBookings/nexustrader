"use server";

import { redirect } from "next/navigation";
import { setOwnerSessionCookie, timingSafePasswordMatches } from "@/lib/auth";

export async function loginAction(formData: FormData) {
  const password = String(formData.get("password") ?? "");
  if (!timingSafePasswordMatches(password)) {
    redirect("/login?error=invalid");
  }
  await setOwnerSessionCookie();
  redirect("/");
}

