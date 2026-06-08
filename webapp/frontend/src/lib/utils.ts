import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, decimals = 4): string {
  return n.toFixed(decimals);
}

export function formatDate(d: string): string {
  return new Date(d).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function analysisDateRange(computedAt: string, windowMonths: number): string {
  const end = new Date(computedAt);
  const start = new Date(computedAt);
  start.setMonth(start.getMonth() - windowMonths);
  const fmt = (d: Date) =>
    d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  return `${fmt(start)} – ${fmt(end)}`;
}

export function halfLife(theta: number): string {
  const days = Math.log(2) / theta;
  if (days > 365) return `${(days / 365).toFixed(1)}y`;
  if (days > 30) return `${(days / 30).toFixed(1)}mo`;
  return `${days.toFixed(1)}d`;
}
