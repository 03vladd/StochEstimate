import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { authApi } from "@/api/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function RegisterPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await authApi.register(email, password);
      setDone(true);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        "Registration failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (done) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background px-6">
        <div className="w-full max-w-xs text-center">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-3">StochEstimate</p>
          <h1 className="text-xl font-semibold mb-2">Check your email</h1>
          <p className="text-sm text-muted-foreground mb-6">
            Verification link sent to <span className="text-foreground font-mono">{email}</span>.
            Click it to activate your account.
          </p>
          <Button variant="outline" className="w-full" onClick={() => navigate("/login")}>
            Go to login
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6">
      <div className="w-full max-w-xs">
        <div className="mb-8">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-3">StochEstimate</p>
          <h1 className="text-xl font-semibold">Create account</h1>
          <p className="text-sm text-muted-foreground mt-1">Start tracking pairs in minutes</p>
        </div>

        <form onSubmit={submit} className="space-y-3">
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Email</label>
            <Input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Password</label>
            <Input
              type="password"
              placeholder="min 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={8}
              required
            />
          </div>
          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">
              {error}
            </p>
          )}
          <Button type="submit" className="w-full mt-1" disabled={loading}>
            {loading ? "Creating account…" : "Create account"}
          </Button>
        </form>

        <p className="text-xs text-muted-foreground text-center mt-6">
          Already have an account?{" "}
          <Link to="/login" className="text-foreground underline underline-offset-2">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
