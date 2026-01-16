"""Navigator LLM helper for path preselection using a repo map."""

from __future__ import annotations

import json
from typing import Iterable

from src.llm.backends.base import GenerationConfig, load_backend

NAVIGATOR_SYSTEM_PROMPT = (
    "You are a codebase navigator.\n"
    "Given a repository map and a user question,\n"
    "identify which files or directories are most likely\n"
    "to contain the information needed to answer the question.\n\n"
    "Do NOT answer the question.\n"
    "Only return a list of file paths or directories to inspect."
)


def _truncate_repo_map(repo_map: dict, max_chars: int) -> str:
    text = json.dumps(repo_map, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "...(truncated)"
    return text


def run_navigator(
    *,
    repo_map: dict,
    question: str,
    backend: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    top_n: int,
    request_id: str | None = None,
    max_map_chars: int = 12000,
) -> list[str]:
    """Use an LLM to produce a shortlist of paths from the repo map."""
    map_text = _truncate_repo_map(repo_map, max_map_chars)
    prompt = (
        f"{NAVIGATOR_SYSTEM_PROMPT}\n\n"
        f"Repository map (JSON):\n{map_text}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Respond with one path per line. Do not add explanations."
    )

    backend_client = load_backend(
        backend,
        model_path=model_id,
        request_id=request_id,
    )
    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
    )
    raw = backend_client.generate(prompt, config)

    paths: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if not cleaned:
            continue
        normalized = cleaned.replace("\\", "/").rstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
        if len(paths) >= top_n:
            break
    return paths
