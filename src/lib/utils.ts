import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function compactNumber(value: number | null | undefined) {
  const numeric = Number(value ?? 0);
  return new Intl.NumberFormat("en-US", {
    notation: Math.abs(numeric) >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: Math.abs(numeric) >= 100 ? 0 : 2,
  }).format(numeric);
}

export function percent(value: number | null | undefined, digits = 2) {
  return `${(Number(value ?? 0) * 100).toFixed(digits)}%`;
}

export function usd(value: number | null | undefined) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: Math.abs(Number(value ?? 0)) >= 100 ? 0 : 2,
  }).format(Number(value ?? 0));
}

export function isoNow() {
  return new Date().toISOString();
}
