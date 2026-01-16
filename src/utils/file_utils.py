"""Utility helpers for repository traversal, filtering, and safe file IO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "migrations",
)

DEFAULT_IGNORE_FILES: tuple[str, ...] = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "poetry.lock",
    "pipfile.lock",
)

DEFAULT_ENCODING_FALLBACKS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str] | None:
    if not extensions:
        return None
    normalized = set()
    for ext in extensions:
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return {ext.lower() for ext in normalized}


def iter_repo_files(
    repo_path: str | Path,
    ignore_dirs: Sequence[str] | None = None,
    allow_extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield files within repo_path applying directory filters and extension allow-list."""

    repo_root = Path(repo_path).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_root}")

    ignore = {d.lower() for d in (ignore_dirs or DEFAULT_IGNORE_DIRS)}
    allow = _normalize_extensions(allow_extensions)
    ignore_files = {name.lower() for name in DEFAULT_IGNORE_FILES}

    for current_dir, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d for d in dirnames if d.lower() not in ignore and not d.startswith(".git")
        ]
        for filename in filenames:
            file_path = Path(current_dir, filename)
            if file_path.name.lower() in ignore_files:
                continue
            if allow and file_path.suffix.lower() not in allow:
                continue
            yield file_path


def is_probably_binary(path: str | Path, sample_bytes: int = 2048) -> bool:
    """Heuristic check for binary files based on null bytes and control character ratio."""

    with Path(path).open("rb") as file_handle:
        chunk = file_handle.read(sample_bytes)
    if b"\x00" in chunk:
        return True
    if not chunk:
        return False

    # consider file binary if >30% bytes are control characters outside whitespace
    text_chars = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
    non_text = chunk.translate(None, text_chars)
    return len(non_text) > len(chunk) * 0.3


def read_text_file(
    path: str | Path,
    encoding_fallbacks: Sequence[str] | None = None,
) -> str:
    """Read text with multiple encodings until one succeeds."""

    encodings = encoding_fallbacks or DEFAULT_ENCODING_FALLBACKS
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise UnicodeDecodeError("unknown", b"", 0, 1, "no encodings provided")
