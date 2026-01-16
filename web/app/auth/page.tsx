'use client';

import { useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function AuthCallbackPage() {
  const router = useRouter();
  const params = useSearchParams();

  useEffect(() => {
    const token = params.get("session_token");
    if (token) {
      window.localStorage.setItem("session_token", token);
      router.replace("/");
      return;
    }
    router.replace("/");
  }, [params, router]);

  return (
    <div className="auth-gate">
      <div className="auth-card stack-sm">
        <h1>Signing you in…</h1>
        <p>Completing GitHub authentication.</p>
      </div>
    </div>
  );
}
