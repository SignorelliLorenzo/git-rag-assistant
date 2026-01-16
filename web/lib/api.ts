import {
  AskRequest,
  AskResponse,
  IndexRequest,
  IndexResponse,
  ListReposResponse,
  LoginRequest,
  LoginResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  {
    method = "GET",
    body,
    token,
  }: {
    method?: string;
    body?: unknown;
    token?: string;
  } = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["x-session-token"] = token;
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const raw = await response.text().catch(() => "");
    const parsed = raw ? (JSON.parse(raw) as unknown) : {};
    const error = (parsed || {}) as Record<string, unknown>;
    const message =
      (error?.error as Record<string, unknown>)?.message ?? response.statusText ?? "Request failed";
    throw new Error(String(message));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const raw = await response.text();
  if (!raw) {
    return undefined as T;
  }
  return JSON.parse(raw) as T;
}

export async function login(payload: LoginRequest): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/auth/login", { method: "POST", body: payload });
}

export async function getSession(token: string): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/auth/session", { token });
}

export async function listRepos(token: string): Promise<ListReposResponse> {
  return apiFetch<ListReposResponse>("/repos", { token });
}

export async function indexRepo(token: string, payload: IndexRequest): Promise<IndexResponse> {
  return apiFetch<IndexResponse>("/index", { method: "POST", body: payload, token });
}

export async function askQuestion(token: string, payload: AskRequest): Promise<AskResponse> {
  return apiFetch<AskResponse>("/ask", { method: "POST", body: payload, token });
}

export async function deleteRepo(token: string, repoId: string): Promise<void> {
  await apiFetch<void>(`/repos/${encodeURIComponent(repoId)}`, {
    method: "DELETE",
    token,
  });
}

export { apiFetch };
