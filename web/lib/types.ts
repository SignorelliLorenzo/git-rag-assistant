export type Provider = "github";

export interface LoginRequest {
  username: string;
  provider?: Provider;
  access_token?: string;
}

export interface LoginResponse {
  session_token: string;
  username: string;
  provider: Provider;
}

export interface RepoInfo {
  repo_id: string;
  repo_path: string;
  indexed: boolean;
  last_indexed_at?: string | null;
  full_name?: string | null;
  repo_url?: string | null;
  default_branch?: string | null;
  provider?: string | null;
}

export interface ListReposResponse {
  repos: RepoInfo[];
}

export interface IndexRequest {
  repo_id: string;
  repo_url?: string;
  branch?: string;
  force_reindex?: boolean;
}

export interface IndexResponse {
  repo_id: string;
  status: "indexed" | "skipped";
  artifacts: {
    repo_path: string;
    chunks: string;
    embeddings: string;
    store_code_dir: string;
    store_structure_dir: string;
  };
  timing_ms: Record<string, number>;
}

export interface AskRequest {
  repo_id: string;
  question: string;
  top_k?: number;
  ensure_index?: boolean;
  backend?: string;
  model_id?: string;
  max_context_chars?: number;
  temperature?: number;
  max_tokens?: number;
  repo_url?: string;
  branch?: string;
}

export interface AskResponse {
  repo_id: string;
  question: string;
  answer: string;
  used_chunks: Array<Record<string, unknown>>;
  model_info: Record<string, unknown>;
  prompt_info: Record<string, unknown>;
  timing_ms: Record<string, number>;
}
