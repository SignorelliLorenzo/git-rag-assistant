"""FastAPI application exposing the RAG pipeline (F6 web backend)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import json
import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode, urlparse
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.loader.chunk_repo import ingest_repository
from src.loader.repo_map import build_repo_map, save_repo_map
from src.embeddings.generate_embeddings import (
    get_model,
    load_chunks,
    embed_chunks,
    write_embeddings,
)
from src.embeddings.vector_store import create_vector_store_from_embeddings
from src.embeddings.retrieval import (
    DEFAULT_MODEL as EMBEDDING_MODEL,
    embed_question,
    retrieve as retrieve_chunks,
)
from src.llm.answer_generation import generate_answer
from src.llm.navigator import run_navigator

from .schemas import (
    AskRequest,
    AskResponse,
    IndexArtifacts,
    IndexRequest,
    IndexResponse,
    RepoInfo,
    ListReposResponse,
    LoginRequest,
    LoginResponse,
)

STRUCTURE_KEYWORDS = {
    "where",
    "location",
    "module",
    "file",
    "directory",
    "path",
    "folder",
    "located",
    "which file",
    "which folder",
}
STRUCTURE_REWRITE_SUFFIX = " file path directory folder module location structure locate definition repository"

REPOS_ROOT = Path("repos").resolve()
EMBEDDINGS_ROOT = Path("embeddings").resolve()
STORE_CODE_DIRNAME = "store_code"
STORE_STRUCTURE_DIRNAME = "store_structure"
REPO_MAP_FILENAME = "repo_map.json"
EMBEDDING_MODEL_INFO_FILENAME = "embedding_model.json"
SESSION_HEADER = "x-session-token"
LOGS_ROOT = Path(os.getenv("RAG_LOG_DIR", "logs")).resolve()
LOG_FILE_PATH = LOGS_ROOT / os.getenv("RAG_LOG_FILE_NAME", "git_rag.log")

LEGACY_EMBEDDING_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
]

_EMBED_MODEL_CACHE: dict[str, str] = {}

ENV_SECRETS_PATH = Path(os.getenv("ENV_SECRETS_PATH", "src/private/env.secrets"))


def _configure_logging() -> None:
    level_name = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=int(os.getenv("RAG_LOG_MAX_BYTES", 10 * 1024 * 1024)),
        backupCount=int(os.getenv("RAG_LOG_BACKUP_COUNT", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    file_handler.setLevel(level)

    has_file_handler = any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == file_handler.baseFilename for h in root_logger.handlers)
    if not has_file_handler:
        root_logger.addHandler(file_handler)

    root_logger.setLevel(level)


_configure_logging()
logger = logging.getLogger("git_rag")


def _load_env_secrets(path: Path) -> dict[str, str]:
    if not path.exists():
        raise RuntimeError(
            f"Secrets file not found at {path}. "
            "Create it with GITHUB_CLIENT_ID/GITHUB_CLIENT_SECRET."
        )
    secrets: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()
    return secrets


_SECRETS = _load_env_secrets(ENV_SECRETS_PATH)
GITHUB_CLIENT_ID = _SECRETS.get("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = _SECRETS.get("GITHUB_CLIENT_SECRET")
APP_BASE_URL = _SECRETS.get("APP_BASE_URL", "http://localhost:3000").strip()
API_BASE_URL = _SECRETS.get("API_BASE_URL", "http://localhost:8000").strip()

if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
    raise RuntimeError(
        "Missing GITHUB_CLIENT_ID or GITHUB_CLIENT_SECRET in secrets file at "
        f"{ENV_SECRETS_PATH}"
    )


@dataclass
class Session:
    token: str
    username: str
    provider: str
    created_at: datetime
    access_token: str | None = None


_SESSIONS: Dict[str, Session] = {}
_OAUTH_STATE: dict[str, datetime] = {}


def _require_session(token: Optional[str] = Header(default=None, alias=SESSION_HEADER)) -> Session:
    if not token or token not in _SESSIONS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid session token")
    return _SESSIONS[token]


def _local_repo_id(full_name: str) -> str:
    return full_name.replace("/", "__")


def _is_github_https_repo(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return host == "github.com"


def _github_token_clone_url(repo_url: str, access_token: str) -> str:
    parsed = urlparse(repo_url)
    if not parsed.scheme or not parsed.netloc:
        return repo_url
    if parsed.scheme != "https":
        return repo_url
    # GitHub supports x-access-token for HTTPS clones.
    return parsed._replace(netloc=f"x-access-token:{access_token}@{parsed.netloc}").geturl()


def _run_git_command(args: list[str], cwd: Optional[Path] = None) -> None:
    process = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Git command failed ({' '.join(args)}): {process.stderr.strip() or process.stdout.strip()}",
        )


def _ensure_repo_checkout(repo_id: str, repo_url: Optional[str], branch: Optional[str]) -> Path:
    REPOS_ROOT.mkdir(parents=True, exist_ok=True)
    repo_path = REPOS_ROOT / repo_id
    if repo_path.exists():
        if branch:
            _run_git_command(["git", "fetch", "--all", "--prune"], cwd=repo_path)
            _run_git_command(["git", "checkout", branch], cwd=repo_path)
            _run_git_command(["git", "pull", "--ff-only"], cwd=repo_path)
        return repo_path

    if not repo_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Repository not found locally. Provide repo_url to clone it.",
        )

    clone_args = ["git", "clone", "--depth", "1"]
    if branch:
        clone_args += ["--branch", branch]
    clone_args += [repo_url, str(repo_path)]
    _run_git_command(clone_args)
    return repo_path


async def _github_get_user(access_token: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get("https://api.github.com/user", headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _github_list_repos(access_token: str) -> list[dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}",
    }
    repos: list[dict] = []
    page = 1
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            resp = await client.get(
                "https://api.github.com/user/repos",
                headers=headers,
                params={"per_page": 100, "page": page, "sort": "updated"},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list) or not batch:
                break
            repos.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    return repos


def _artifact_paths(repo_id: str) -> tuple[Path, Path, Path, Path]:
    base = EMBEDDINGS_ROOT / repo_id
    chunks = base / "chunks.json"
    embeddings = base / "embeddings.json"
    store_code = base / STORE_CODE_DIRNAME
    store_structure = base / STORE_STRUCTURE_DIRNAME
    return chunks, embeddings, store_code, store_structure


def _repo_map_path(repo_id: str) -> Path:
    return EMBEDDINGS_ROOT / repo_id / REPO_MAP_FILENAME


def _embedding_model_info_path(repo_id: str) -> Path:
    return EMBEDDINGS_ROOT / repo_id / EMBEDDING_MODEL_INFO_FILENAME


def _save_embedding_model_info(repo_id: str, model_name: str) -> None:
    info_path = _embedding_model_info_path(repo_id)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": model_name, "saved_at": datetime.utcnow().isoformat()}
    with info_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    _EMBED_MODEL_CACHE[repo_id] = model_name


def _resolve_embedding_model(repo_id: str) -> str:
    if repo_id in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[repo_id]

    info_path = _embedding_model_info_path(repo_id)
    if info_path.exists():
        try:
            with info_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            model_name = str(payload.get("model_name") or "").strip()
            if model_name:
                _EMBED_MODEL_CACHE[repo_id] = model_name
                return model_name
        except Exception:  # noqa: BLE001
            logger.exception("embedding_model_info_read_failed repo_id=%s", repo_id)

    fallback = os.getenv("RAG_LEGACY_EMBED_MODEL", LEGACY_EMBEDDING_MODELS[0])
    logger.warning(
        "embedding_model_metadata_missing repo_id=%s fallback=%s",
        repo_id,
        fallback,
    )
    _EMBED_MODEL_CACHE[repo_id] = fallback
    return fallback


def _load_repo_map(repo_id: str, repo_url: Optional[str], branch: Optional[str], session: Session) -> dict:
    map_path = _repo_map_path(repo_id)
    repo_path = REPOS_ROOT / repo_id

    # Ensure repo checkout if map is missing and a repo_url is provided.
    if not map_path.exists():
        if repo_url and session.provider == "github" and session.access_token and _is_github_https_repo(str(repo_url)):
            repo_url = _github_token_clone_url(str(repo_url), session.access_token)
        if not repo_path.exists() and repo_url:
            repo_path = _ensure_repo_checkout(repo_id, repo_url, branch)
        if not repo_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository not found locally for repo_id={repo_id}. Provide repo_url to clone.",
            )
        repo_map = build_repo_map(repo_path)
        save_repo_map(repo_map, map_path)
        return repo_map

    try:
        with map_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        # Rebuild if corrupted.
        if repo_path.exists():
            repo_map = build_repo_map(repo_path)
            save_repo_map(repo_map, map_path)
            return repo_map
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read repo map for repo_id={repo_id}: {exc}",
        ) from exc


def _format_sources_block(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["```sources"]
    for chunk in chunks:
        path = chunk.get("file_path", "unknown")
        start = chunk.get("start_line")
        end = chunk.get("end_line")
        label = f"{path}"
        if isinstance(start, int) and isinstance(end, int):
            label += f" (lines {start}-{end})"
        lines.append(label)
        text = str(chunk.get("text", "")).strip()
        if text:
            # Limit extremely long snippets in the appended block.
            snippet = text if len(text) <= 2000 else text[:2000] + "\n... [truncated]"
            lines.append(snippet)
        lines.append("")  # blank line between sources
    lines.append("```")
    return "\n".join(lines).strip()


def _artifacts_exist(repo_id: str) -> bool:
    chunks, embeddings, store_code, store_structure = _artifact_paths(repo_id)
    return (
        chunks.exists()
        and embeddings.exists()
        and (store_code / "store.json").exists()
        and (store_structure / "store.json").exists()
    )


def _is_structure_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in STRUCTURE_KEYWORDS)


def _rewrite_structure_question(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return STRUCTURE_REWRITE_SUFFIX.strip()
    return f"{stripped} {STRUCTURE_REWRITE_SUFFIX}"


def _run_index_pipeline(repo_id: str, repo_path: Path) -> tuple[IndexArtifacts, dict[str, float]]:
    EMBEDDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    chunks_path, embeddings_path, store_code_dir, store_structure_dir = _artifact_paths(repo_id)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    ingest_repository(repo_path, chunks_path, repo_id=repo_id)
    timings["ingest_ms"] = (time.perf_counter() - t0) * 1000

    chunk_records = load_chunks(chunks_path)

    t0 = time.perf_counter()
    model = get_model(EMBEDDING_MODEL, device="cpu")
    embeddings = None
    last_error: RuntimeError | None = None
    for batch_size in (32, 16, 8, 4, 2, 1):
        try:
            embeddings = embed_chunks(chunk_records, model, batch_size=batch_size)
            last_error = None
            break
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "CUDA out of memory" in msg
                or "DefaultCPUAllocator" in msg
                or "not enough memory" in msg
                or "out of memory" in msg
            ):
                last_error = exc
                logger.exception(
                    "index_embed_oom_retry request_id=n/a repo_id=%s batch_size=%s",
                    repo_id,
                    batch_size,
                )
                continue
            raise

    if embeddings is None:
        raise last_error if last_error else RuntimeError("Embedding generation failed.")

    write_embeddings(chunk_records, embeddings, embeddings_path)
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000
    _save_embedding_model_info(repo_id, EMBEDDING_MODEL)

    t0 = time.perf_counter()
    code_store, _summary_code = create_vector_store_from_embeddings(
        repo_id=repo_id,
        embeddings_path=embeddings_path,
        backend_preference="faiss",
        metric="cosine",
        normalize=True,
        chunk_types=["code"],
    )
    code_store.save(store_code_dir)
    timings["vector_store_code_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    structure_store, _summary_structure = create_vector_store_from_embeddings(
        repo_id=repo_id,
        embeddings_path=embeddings_path,
        backend_preference="faiss",
        metric="cosine",
        normalize=True,
        chunk_types=["file_summary", "directory_summary"],
    )
    structure_store.save(store_structure_dir)
    timings["vector_store_structure_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    repo_map = build_repo_map(repo_path)
    save_repo_map(repo_map, _repo_map_path(repo_id))
    timings["repo_map_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())

    artifacts = IndexArtifacts(
        repo_path=str(repo_path),
        chunks=str(chunks_path),
        embeddings=str(embeddings_path),
        store_code_dir=str(store_code_dir),
        store_structure_dir=str(store_structure_dir),
    )
    return artifacts, timings


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        logger.warning("delete_repo_cleanup_failed path=%s", path, exc_info=exc_info)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        try:
            path.unlink()
            return
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
            return
    shutil.rmtree(path, onerror=_handle_remove_readonly)


def _delete_repo_disk(repo_id: str) -> None:
    repo_path = REPOS_ROOT / repo_id
    embed_root = EMBEDDINGS_ROOT / repo_id
    logger.info("delete_repo_disk repo_id=%s repo_path=%s", repo_id, repo_path)
    for path in [embed_root, repo_path]:
        _remove_path(path)
    _EMBED_MODEL_CACHE.pop(repo_id, None)


def _list_repositories() -> list[RepoInfo]:
    if not REPOS_ROOT.exists():
        return []
    repos: list[RepoInfo] = []
    for folder in sorted(REPOS_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        repo_id = folder.name
        store_manifest = EMBEDDINGS_ROOT / repo_id / STORE_CODE_DIRNAME / "store.json"
        indexed = store_manifest.exists()
        last_indexed = datetime.fromtimestamp(store_manifest.stat().st_mtime) if indexed else None
        repos.append(
            RepoInfo(
                repo_id=repo_id,
                repo_path=str(folder),
                indexed=indexed,
                last_indexed_at=last_indexed,
            )
        )
    return repos


app = FastAPI(title="GitRepo RAG Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        duration_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        getattr(response, "status_code", "n/a"),
        duration_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "http_error request_id=%s method=%s path=%s status=%s detail=%s",
        request_id,
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "http_error",
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "unhandled_error request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "type": exc.__class__.__name__,
                "request_id": request_id,
            }
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/login")
def github_login() -> RedirectResponse:
    state = uuid4().hex
    _OAUTH_STATE[state] = datetime.utcnow()
    query = urlencode(
        {
            "client_id": GITHUB_CLIENT_ID,
            "redirect_uri": f"{API_BASE_URL}/auth/callback",
            "state": state,
            "scope": "repo",
        }
    )
    url = f"https://github.com/login/oauth/authorize?{query}"
    return RedirectResponse(url=url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@app.get("/auth/callback")
async def github_callback(code: str = Query(...), state: str = Query(...)) -> RedirectResponse:
    if state not in _OAUTH_STATE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")
    _OAUTH_STATE.pop(state, None)

    headers = {"Accept": "application/json"}
    data = {
        "client_id": GITHUB_CLIENT_ID,
        "client_secret": GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": f"{API_BASE_URL}/auth/callback",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            headers=headers,
            data=data,
        )
        raw_text = token_resp.text
        try:
            token_resp.raise_for_status()
            token_json = token_resp.json()
        except Exception as exc:  # noqa: BLE001
            logging.error("GitHub token exchange error: status=%s body=%s", token_resp.status_code, raw_text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OAuth token exchange failed (HTTP error)",
            ) from exc

    access_token = token_json.get("access_token")
    if not access_token:
        logging.error("GitHub token exchange missing access_token: body=%s", raw_text)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth token exchange failed (no access_token)",
        )

    user = await _github_get_user(access_token)
    username = str(user.get("login") or "github-user")
    session_token = uuid4().hex

    _SESSIONS[session_token] = Session(
        token=session_token,
        username=username,
        provider="github",
        created_at=datetime.utcnow(),
        access_token=access_token,
    )

    return RedirectResponse(
        url=f"{APP_BASE_URL.rstrip('/')}/auth?session_token={session_token}",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    token = uuid4().hex
    session = Session(
        token=token,
        username=payload.username,
        provider=payload.provider,
        created_at=datetime.utcnow(),
    )
    _SESSIONS[token] = session
    return LoginResponse(session_token=token, username=session.username, provider=session.provider)


@app.get("/auth/session", response_model=LoginResponse)
def auth_session(session: Session = Depends(_require_session)) -> LoginResponse:
    return LoginResponse(session_token=session.token, username=session.username, provider=session.provider)


@app.get("/repos", response_model=ListReposResponse)
async def list_repos(session: Session = Depends(_require_session)) -> ListReposResponse:
    if session.provider != "github" or not session.access_token:
        return ListReposResponse(repos=_list_repositories())

    gh_repos = await _github_list_repos(session.access_token)
    repos: list[RepoInfo] = []
    for repo in gh_repos:
        full_name = str(repo.get("full_name") or "")
        if not full_name:
            continue
        repo_id = _local_repo_id(full_name)
        clone_url = str(repo.get("clone_url") or "")
        default_branch = str(repo.get("default_branch") or "")

        repo_path = (REPOS_ROOT / repo_id)
        store_manifest = EMBEDDINGS_ROOT / repo_id / STORE_CODE_DIRNAME / "store.json"
        indexed = store_manifest.exists()
        last_indexed = datetime.fromtimestamp(store_manifest.stat().st_mtime) if indexed else None

        repos.append(
            RepoInfo(
                repo_id=repo_id,
                repo_path=str(repo_path) if repo_path.exists() else "",
                indexed=indexed,
                last_indexed_at=last_indexed,
                full_name=full_name,
                repo_url=clone_url,
                default_branch=default_branch,
                provider="github",
            )
        )
    return ListReposResponse(repos=repos)


@app.get("/repos/{repo_id}/map")
async def get_repo_map(
    repo_id: str,
    request: Request,
    repo_url: Optional[str] = Query(default=None),
    branch: Optional[str] = Query(default=None),
    session: Session = Depends(_require_session),
) -> dict:
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "repo_map_request request_id=%s repo_id=%s has_repo_url=%s branch=%s",
        request_id,
        repo_id,
        bool(repo_url),
        branch,
    )
    repo_map = _load_repo_map(repo_id, repo_url, branch, session)
    logger.info(
        "repo_map_served request_id=%s repo_id=%s node_count=%s",
        request_id,
        repo_id,
        len(repo_map.get("root", {}).get("children", []) or []),
    )
    return repo_map


@app.delete("/repos/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repo(repo_id: str, session: Session = Depends(_require_session)) -> None:
    # Allow deletion of local artifacts; no provider access needed.
    _delete_repo_disk(repo_id)
    return None


@app.post("/index", response_model=IndexResponse)
def index_repository(
    payload: IndexRequest,
    request: Request,
    session: Session = Depends(_require_session),
) -> IndexResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "index_start request_id=%s repo_id=%s force_reindex=%s has_repo_url=%s branch=%s",
        request_id,
        payload.repo_id,
        payload.force_reindex,
        bool(payload.repo_url),
        payload.branch,
    )
    repo_url = payload.repo_url
    if (
        repo_url
        and session.provider == "github"
        and session.access_token
        and _is_github_https_repo(str(repo_url))
    ):
        repo_url = _github_token_clone_url(str(repo_url), session.access_token)

    repo_path = _ensure_repo_checkout(payload.repo_id, repo_url, payload.branch)
    artifacts_exist = _artifacts_exist(payload.repo_id)
    if artifacts_exist and not payload.force_reindex:
        logger.info(
            "index_skip request_id=%s repo_id=%s reason=artifacts_exist",
            request_id,
            payload.repo_id,
        )
        chunks_path, embeddings_path, store_code_dir, store_structure_dir = _artifact_paths(payload.repo_id)
        artifacts = IndexArtifacts(
            repo_path=str(repo_path),
            chunks=str(chunks_path),
            embeddings=str(embeddings_path),
            store_code_dir=str(store_code_dir),
            store_structure_dir=str(store_structure_dir),
        )
        return IndexResponse(
            repo_id=payload.repo_id,
            status="skipped",
            artifacts=artifacts,
            timing_ms={"total_ms": 0.0},
        )

    t0 = time.perf_counter()
    artifacts, timings = _run_index_pipeline(payload.repo_id, repo_path)
    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "index_complete request_id=%s repo_id=%s total_ms=%.2f ingest_ms=%.2f embed_ms=%.2f vs_code_ms=%.2f vs_structure_ms=%.2f",
        request_id,
        payload.repo_id,
        total_ms,
        timings.get("ingest_ms", 0.0),
        timings.get("embed_ms", 0.0),
        timings.get("vector_store_code_ms", 0.0),
        timings.get("vector_store_structure_ms", 0.0),
    )
    return IndexResponse(repo_id=payload.repo_id, status="indexed", artifacts=artifacts, timing_ms=timings)


@app.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    request: Request,
    session: Session = Depends(_require_session),
) -> AskResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "ask_start request_id=%s repo_id=%s top_k=%s ensure_index=%s backend=%s model_id=%s",
        request_id,
        payload.repo_id,
        payload.top_k,
        payload.ensure_index,
        payload.backend,
        payload.model_id,
    )
    logger.info(
        "ask_question_text request_id=%s repo_id=%s question=%s",
        request_id,
        payload.repo_id,
        payload.question.strip(),
    )
    store_code_dir = EMBEDDINGS_ROOT / payload.repo_id / STORE_CODE_DIRNAME
    store_structure_dir = EMBEDDINGS_ROOT / payload.repo_id / STORE_STRUCTURE_DIRNAME
    code_manifest = store_code_dir / "store.json"
    structure_manifest = store_structure_dir / "store.json"
    repo_path = REPOS_ROOT / payload.repo_id

    if payload.ensure_index or not (code_manifest.exists() and structure_manifest.exists()):
        logger.info(
            "ask_ensure_index request_id=%s repo_id=%s",
            request_id,
            payload.repo_id,
        )
        repo_url = payload.repo_url
        if (
            repo_url
            and session.provider == "github"
            and session.access_token
            and _is_github_https_repo(str(repo_url))
        ):
            repo_url = _github_token_clone_url(str(repo_url), session.access_token)

        repo_path = _ensure_repo_checkout(payload.repo_id, repo_url, payload.branch)
        t0 = time.perf_counter()
        _run_index_pipeline(payload.repo_id, repo_path)
        logger.info(
            "ask_index_built request_id=%s repo_id=%s duration_ms=%.2f",
            request_id,
            payload.repo_id,
            (time.perf_counter() - t0) * 1000,
        )
        code_manifest = store_code_dir / "store.json"
        structure_manifest = store_structure_dir / "store.json"

    if not code_manifest.exists() or not structure_manifest.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store not found for repo_id={payload.repo_id}. Run indexing first.",
        )

    routing_info = {
        "index": "guided",
        "stores": ["code", "structure"],
    }
    timings: dict[str, float] = {}

    navigator_paths: list[str] | None = None
    if payload.use_navigator:
        try:
            repo_map = _load_repo_map(payload.repo_id, payload.repo_url, payload.branch, session)
            t_nav = time.perf_counter()
            navigator_paths = run_navigator(
                repo_map=repo_map,
                question=payload.question,
                backend=payload.navigator_backend,
                model_id=payload.navigator_model_id,
                temperature=payload.navigator_temperature,
                max_tokens=payload.navigator_max_tokens,
                top_n=payload.navigator_top_n,
                request_id=request_id,
            )
            timings["navigator_ms"] = (time.perf_counter() - t_nav) * 1000
            logger.info(
                "ask_navigator_done request_id=%s repo_id=%s candidates=%s navigator_ms=%.2f",
                request_id,
                payload.repo_id,
                len(navigator_paths or []),
                timings["navigator_ms"],
            )
            logger.info(
                "ask_navigator_paths request_id=%s repo_id=%s paths=%s",
                request_id,
                payload.repo_id,
                navigator_paths,
            )
        except Exception:
            logger.exception("ask_navigator_failed request_id=%s repo_id=%s", request_id, payload.repo_id)
            navigator_paths = None

    retrieval_model = _resolve_embedding_model(payload.repo_id)
    retrieval_start = time.perf_counter()
    embed_start = time.perf_counter()
    query_vector = embed_question(payload.question, retrieval_model, payload.device)
    query_embed_ms = (time.perf_counter() - embed_start) * 1000
    logger.info(
        "ask_query_embedding request_id=%s repo_id=%s model=%s embed_ms=%.2f",
        request_id,
        payload.repo_id,
        retrieval_model,
        query_embed_ms,
    )

    try:
        hits_code = retrieve_chunks(
            repo_id=payload.repo_id,
            question=payload.question,
            top_k=payload.top_k,
            store_dir=store_code_dir,
            include_text=True,
            path_filters=navigator_paths,
            request_id=request_id,
            query_vector=query_vector,
            query_vector_ms=query_embed_ms,
            model_name=retrieval_model,
            device=payload.device,
        )
        hits_structure = retrieve_chunks(
            repo_id=payload.repo_id,
            question=payload.question,
            top_k=payload.top_k,
            store_dir=store_structure_dir,
            include_text=True,
            path_filters=navigator_paths,
            request_id=request_id,
            query_vector=query_vector,
            query_vector_ms=0.0,
             model_name=retrieval_model,
             device=payload.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    for h in hits_code:
        h.setdefault("retrieval_store", "code")
    for h in hits_structure:
        h.setdefault("retrieval_store", "structure")

    # Merge and select top_k by score.
    merged_hits = list(hits_code) + list(hits_structure)
    merged_hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    hits = merged_hits[: int(payload.top_k)]

    timings["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
    logger.info(
        "ask_retrieval_done request_id=%s repo_id=%s hits=%s hits_code=%s hits_structure=%s retrieval_ms=%.2f path_filters=%s",
        request_id,
        payload.repo_id,
        len(hits),
        len(hits_code),
        len(hits_structure),
        timings["retrieval_ms"],
        bool(navigator_paths),
    )

    t0 = time.perf_counter()
    answer = generate_answer(
        repo_id=payload.repo_id,
        question=payload.question,
        retrieval_hits=hits,
        backend=payload.backend,
        model_id=payload.model_id,
        ollama_host=payload.ollama_host,
        top_k=payload.top_k,
        max_context_chars=payload.max_context_chars,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        device=payload.device,
        model_type=payload.model_type,
        gpu_layers=payload.gpu_layers,
        n_threads=payload.n_threads,
        request_id=request_id,
    )
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    logger.info(
        "ask_generation_done request_id=%s repo_id=%s generation_ms=%.2f total_ms=%.2f",
        request_id,
        payload.repo_id,
        timings["generation_ms"],
        timings["total_ms"],
    )

    prompt_info = dict(answer["prompt_info"])
    prompt_info["retrieval_routing"] = routing_info
    prompt_info["navigator_used"] = bool(navigator_paths)

    sources_block = _format_sources_block(answer["used_chunks"])
    final_answer = f"{answer['answer']}\n\n{sources_block}" if sources_block else answer["answer"]

    logger.info(
        "ask_answer_text request_id=%s repo_id=%s answer=%s",
        request_id,
        payload.repo_id,
        final_answer,
    )
    logger.debug(
        "ask_answer_chunks request_id=%s repo_id=%s used_chunks=%s",
        request_id,
        payload.repo_id,
        answer["used_chunks"],
    )

    return AskResponse(
        repo_id=answer["repo_id"],
        question=answer["question"],
        answer=final_answer,
        used_chunks=answer["used_chunks"],
        model_info=answer["model_info"],
        prompt_info=prompt_info,
        timing_ms=timings,
    )
