import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "@/store/auth";
import { ToastProvider } from "@/components/ui/toast";
import CommandPalette from "@/components/ui/command-palette";
import LoginPage from "@/pages/LoginPage";
import RegisterPage from "@/pages/RegisterPage";
import VerifyEmailPage from "@/pages/VerifyEmailPage";
import DashboardPage from "@/pages/DashboardPage";
import PairDetailPage from "@/pages/PairDetailPage";
import type { ReactNode } from "react";

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  return (
    <>
      {children}
      <CommandPalette />
    </>
  );
}

function PublicOnly({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  if (user) return <Navigate to="/dashboard" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route
              path="/login"
              element={
                <PublicOnly>
                  <LoginPage />
                </PublicOnly>
              }
            />
            <Route
              path="/register"
              element={
                <PublicOnly>
                  <RegisterPage />
                </PublicOnly>
              }
            />
            <Route path="/verify-email" element={<VerifyEmailPage />} />
            <Route
              path="/dashboard"
              element={
                <RequireAuth>
                  <DashboardPage />
                </RequireAuth>
              }
            />
            <Route
              path="/pairs/:pairId"
              element={
                <RequireAuth>
                  <PairDetailPage />
                </RequireAuth>
              }
            />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </BrowserRouter>
      </ToastProvider>
    </AuthProvider>
  );
}
