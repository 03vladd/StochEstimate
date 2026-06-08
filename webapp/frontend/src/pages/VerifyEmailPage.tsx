import { useEffect, useRef, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import { Button } from "@/components/ui/button";

export default function VerifyEmailPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const called = useRef(false);

  useEffect(() => {
    if (called.current) return;
    called.current = true;
    const token = params.get("token");
    if (!token) { setStatus("error"); return; }
    api
      .get(`/auth/verify?token=${token}`)
      .then(() => setStatus("ok"))
      .catch(() => setStatus("error"));
  }, [params]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6">
      <div className="w-full max-w-xs text-center">
        <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-3">StochEstimate</p>
        <h1 className="text-xl font-semibold mb-2">
          {status === "loading" && "Verifying…"}
          {status === "ok" && "Email verified"}
          {status === "error" && "Verification failed"}
        </h1>
        <p className="text-sm text-muted-foreground mb-6">
          {status === "loading" && "Please wait."}
          {status === "ok" && "Your account is active. You can now sign in."}
          {status === "error" && "The link may have expired or already been used."}
        </p>
        {status !== "loading" && (
          <Button className="w-full" onClick={() => navigate("/login")}>
            Go to login
          </Button>
        )}
      </div>
    </div>
  );
}
