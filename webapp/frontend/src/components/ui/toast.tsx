import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export type ToastVariant = "success" | "error" | "info" | "warning";

interface ToastItem {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
  duration?: number;
}

interface ToastContextType {
  toast: (item: Omit<ToastItem, "id">) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

const LEFT_BORDER: Record<ToastVariant, string> = {
  success: "border-l-green-500",
  error:   "border-l-red-500",
  warning: "border-l-amber-500",
  info:    "border-l-blue-500/70",
};

const PROGRESS_COLOR: Record<ToastVariant, string> = {
  success: "bg-green-500",
  error:   "bg-red-500",
  warning: "bg-amber-500",
  info:    "bg-blue-500/70",
};

function ToastItem({ item, onRemove }: { item: ToastItem; onRemove: () => void }) {
  const duration = item.duration ?? 4000;

  useEffect(() => {
    const t = setTimeout(onRemove, duration);
    return () => clearTimeout(t);
  }, [duration, onRemove]);

  return (
    <div
      className={cn(
        "relative w-72 rounded-lg border border-border/80 border-l-2 bg-card overflow-hidden",
        "shadow-xl shadow-black/40",
        LEFT_BORDER[item.variant]
      )}
    >
      <div className="px-4 py-3 pr-9">
        <p className="text-sm font-medium leading-snug">{item.title}</p>
        {item.description && (
          <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{item.description}</p>
        )}
      </div>
      <button
        onClick={onRemove}
        className="absolute top-2.5 right-2.5 text-muted-foreground/30 hover:text-muted-foreground/70 transition-colors"
      >
        <X className="h-3.5 w-3.5" />
      </button>
      {/* Depleting progress bar */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-border/40">
        <div
          className={cn("h-full origin-left", PROGRESS_COLOR[item.variant])}
          style={{ animation: `toast-shrink ${duration}ms linear forwards` }}
        />
      </div>
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((item: Omit<ToastItem, "id">) => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev.slice(-4), { ...item, id }]);
  }, []);

  const remove = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 items-end pointer-events-none">
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto">
            <ToastItem item={t} onRemove={() => remove(t.id)} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
