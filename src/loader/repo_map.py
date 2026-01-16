"""Build a compressed semantic repository map (lightweight tree)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

from src.utils.file_utils import DEFAULT_IGNORE_DIRS


@dataclass(frozen=True)
class RepoMapNode:
    path: str
    type: str  # "dir" | "file" | "ellipsis"
    language: str | None = None
    size_bytes: int | None = None
    children: List["RepoMapNode"] | None = None
    counts: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "path": self.path,
            "type": self.type,
        }
        if self.language:
            payload["language"] = self.language
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.children is not None:
            payload["children"] = [child.to_dict() for child in self.children]
        if self.counts:
            payload["counts"] = self.counts
        return payload


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
}


def _iter_entries(path: Path) -> Iterator[Path]:
    try:
        yield from path.iterdir()
    except FileNotFoundError:
        return


def _build_tree(
    root: Path,
    current: Path,
    *,
    depth: int,
    max_depth: int,
    max_children: int,
    ignore_dirs: set[str],
) -> RepoMapNode | None:
    rel_path = current.relative_to(root).as_posix() or "."

    if current.is_dir():
        if depth > max_depth:
            return None
        children: list[RepoMapNode] = []
        entries = []
        for entry in _iter_entries(current):
            if entry.name in ignore_dirs:
                continue
            entries.append(entry)
        entries.sort(key=lambda p: p.name.lower())

        dir_count = 0
        file_count = 0
        for entry in entries[:max_children]:
            node = _build_tree(
                root,
                entry,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                ignore_dirs=ignore_dirs,
            )
            if node:
                children.append(node)
                if node.type == "dir":
                    dir_count += 1
                elif node.type == "file":
                    file_count += 1

        remaining = len(entries) - len(children)
        if remaining > 0:
            children.append(
                RepoMapNode(
                    path=f"{rel_path}/...",
                    type="ellipsis",
                    counts={"remaining": remaining},
                )
            )

        return RepoMapNode(
            path=rel_path,
            type="dir",
            children=children,
            counts={"dirs": dir_count, "files": file_count},
        )

    if current.is_file():
        ext = current.suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext)
        try:
            size_bytes = current.stat().st_size
        except OSError:
            size_bytes = None
        return RepoMapNode(
            path=rel_path,
            type="file",
            language=language,
            size_bytes=size_bytes,
        )

    return None


def build_repo_map(
    repo_root: Path,
    *,
    max_depth: int = 6,
    max_children: int = 40,
    ignore_dirs: set[str] | None = None,
) -> dict:
    repo_root = repo_root.resolve()
    ignore = set(ignore_dirs) if ignore_dirs else set(DEFAULT_IGNORE_DIRS)
    tree = _build_tree(
        repo_root,
        repo_root,
        depth=0,
        max_depth=max_depth,
        max_children=max_children,
        ignore_dirs=ignore,
    )
    return {
        "schema_version": 1,
        "repo_id": repo_root.name,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "root": tree.to_dict() if tree else {},
    }


def save_repo_map(map_data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(map_data, fh, ensure_ascii=False, indent=2)


def flatten_paths(map_data: dict) -> list[str]:
    """Return all paths (files/dirs) from a repo map."""

    def _walk(node: dict, acc: list[str]) -> None:
        path = node.get("path")
        ntype = node.get("type")
        if path and ntype in {"dir", "file"}:
            acc.append(path)
        for child in node.get("children", []) or []:
            _walk(child, acc)

    acc: list[str] = []
    root = map_data.get("root") or {}
    _walk(root, acc)
    return acc
