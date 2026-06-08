import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/store/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      navigate("/dashboard");
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        "Login failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6">
      <div className="w-full max-w-xs">
        <div className="mb-8">
          <p className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-3">StochEstimate</p>
          <h1 className="text-xl font-semibold">Sign in</h1>
          <p className="text-sm text-muted-foreground mt-1">Access your pair watchlist</p>
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
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-md px-3 py-2">
              {error}
            </p>
          )}
          <Button type="submit" className="w-full mt-1" disabled={loading}>
            {loading ? "Signing in…" : "Sign in"}
          </Button>
        </form>

        <p className="text-xs text-muted-foreground text-center mt-6">
          No account?{" "}
          <Link to="/register" className="text-foreground underline underline-offset-2">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
