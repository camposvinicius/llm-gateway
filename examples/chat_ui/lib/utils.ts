import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function usdFromMicroUsd(microUsd?: number) {
  if (typeof microUsd !== "number") return "$0.000000";
  return `$${(microUsd / 1_000_000).toFixed(6)}`;
}

export function compactNumber(value?: number) {
  if (typeof value !== "number") return "0";
  return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}