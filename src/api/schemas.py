"""Pydantic schemas for the F6 web application API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, PositiveInt


class LoginRequest(BaseModel):
    """Simple login payload (placeholder for full OAuth flow)."""

    username: str = Field(..., description="Display name for the local session.")
    provider: Literal["github"] = Field("github", description="Git provider identifier.")
    access_token: Optional[str] = Field(
        default=None,
        description="Optional GitHub access token (not persisted in MVP).",
    )


class LoginResponse(BaseModel):
    session_token: str
    username: str
    provider: str


class RepoInfo(BaseModel):
    repo_id: str
    repo_path: str
    indexed: bool
    last_indexed_at: Optional[datetime] = None
    full_name: Optional[str] = None
    repo_url: Optional[str] = None
    default_branch: Optional[str] = None
    provider: Optional[str] = None


class ListReposResponse(BaseModel):
    repos: list[RepoInfo]


class IndexRequest(BaseModel):
    repo_id: str = Field(..., description="Local identifier for the repository.")
    repo_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional remote Git URL. If provided and repo is missing, it will be cloned.",
    )
    branch: Optional[str] = Field(default=None, description="Optional branch to checkout when cloning.")
    force_reindex: bool = False


class IndexArtifacts(BaseModel):
    repo_path: str
    chunks: str
    embeddings: str
    store_code_dir: str
    store_structure_dir: str


class IndexResponse(BaseModel):
    repo_id: str
    status: Literal["indexed", "skipped"]
    artifacts: IndexArtifacts
    timing_ms: dict[str, float]


class AskRequest(BaseModel):
    repo_id: str
    question: str = Field(..., min_length=4)
    top_k: PositiveInt = Field(5, description="Number of chunks to retrieve.")
    ensure_index: bool = False
    backend: str = "ollama"
    model_id: str = "qwen2.5:7b"
    use_navigator: bool = True
    navigator_backend: str = "ollama"
    navigator_model_id: str = "qwen2.5:7b"
    navigator_temperature: float = Field(0.0, ge=0.0, le=2.0)
    navigator_max_tokens: PositiveInt = 128
    navigator_top_n: PositiveInt = 12
    max_context_chars: PositiveInt = 12_000
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: PositiveInt = 512
    repo_url: Optional[HttpUrl] = None
    branch: Optional[str] = None
    device: Optional[str] = None
    model_type: Optional[str] = None
    gpu_layers: Optional[int] = None
    n_threads: Optional[int] = None
    ollama_host: Optional[str] = None


class AskResponse(BaseModel):
    repo_id: str
    question: str
    answer: str
    used_chunks: list[dict]
    model_info: dict
    prompt_info: dict
    timing_ms: dict[str, float]


class ErrorResponse(BaseModel):
    error: dict
