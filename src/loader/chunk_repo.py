"""Repository ingestion CLI for chunking source files into embedding-friendly docs (F1)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator, Sequence

from src.utils.file_utils import (
    DEFAULT_ENCODING_FALLBACKS,
    DEFAULT_IGNORE_DIRS,
    is_probably_binary,
    iter_repo_files,
    read_text_file,
)


@dataclass
class Chunk:
    repo_id: str
    file_path: str
    chunk_id: int
    text: str
    start_line: int | None
    end_line: int | None
    chunk_type: str = "code"
    dir_path: str | None = None
    language: str | None = None
    embedding_text: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


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


def detect_language(file_path: Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), "unknown")


def chunk_lines(
    text: str,
    max_lines: int,
    overlap: int = 0,
) -> Iterator[tuple[str, int | None, int | None]]:
    """Yield (chunk_text, start_line, end_line) tuples using line-based chunking."""

    if max_lines <= 0:
        raise ValueError("max_lines must be positive")
    if overlap >= max_lines:
        raise ValueError("overlap must be smaller than max_lines")

    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        yield ("", None, None)
        return

    start_index = 0
    chunk_idx = 0
    while start_index < total_lines:
        end_index = min(start_index + max_lines, total_lines)
        chunk_text = "\n".join(lines[start_index:end_index])
        start_line = start_index + 1
        end_line = end_index
        yield (chunk_text, start_line, end_line)
        chunk_idx += 1
        if end_index >= total_lines:
            break
        start_index = end_index - overlap if overlap else end_index


def _format_embedding_text(
    *,
    path: str,
    dir_path: str,
    language: str,
    chunk_type: str,
    start_line: int | None,
    end_line: int | None,
    body: str,
) -> str:
    header = [
        f"TYPE: {chunk_type}",
        f"PATH: {path}",
        f"DIR: {dir_path or '.'}",
        f"LANG: {language}",
    ]
    if start_line and end_line:
        header.append(f"LINES: {start_line}-{end_line}")
    header.append("-----")
    return "\n".join(header + [body.strip()])


def chunk_file(
    repo_id: str,
    file_path: Path,
    rel_path: str,
    max_lines: int,
    overlap: int,
    encoding_fallbacks: Sequence[str],
) -> tuple[list[Chunk], dict]:
    """Convert a single file into a list of chunks."""

    text = read_text_file(file_path, encoding_fallbacks=encoding_fallbacks)
    chunks: list[Chunk] = []
    dir_path = file_path.parent.as_posix()
    if dir_path == ".":
        dir_path = ""
    language = detect_language(file_path)
    file_summary_info: dict = {
        "imports": [],
        "symbols": [],
        "module": file_path.stem,
        "dir_path": dir_path,
        "language": language,
    }

    imports: list[str] = []
    symbols: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
        elif re.match(r"^(class|def|function)\s", stripped):
            symbols.append(stripped.split("(")[0])
    file_summary_info["imports"] = imports[:20]
    file_summary_info["symbols"] = symbols[:30]

    for chunk_idx, (chunk_text, start_line, end_line) in enumerate(
        chunk_lines(text, max_lines=max_lines, overlap=overlap)
    ):
        if not chunk_text.strip():
            continue
        chunks.append(
            Chunk(
                repo_id=repo_id,
                file_path=rel_path,
                chunk_id=chunk_idx,
                text=chunk_text,
                start_line=start_line,
                end_line=end_line,
                chunk_type="code",
                dir_path=dir_path,
                language=language,
                embedding_text=_format_embedding_text(
                    path=rel_path,
                    dir_path=dir_path or ".",
                    language=language,
                    chunk_type="code",
                    start_line=start_line,
                    end_line=end_line,
                    body=chunk_text,
                ),
            )
        )
    return chunks, file_summary_info


def _build_file_summary_chunk(
    repo_id: str,
    rel_path: str,
    dir_path: str,
    language: str,
    summary_info: dict,
) -> Chunk:
    summary_lines = [
        f"File: {rel_path}",
        f"Directory: {dir_path or '.'}",
        f"Language: {language}",
    ]
    if summary_info.get("imports"):
        summary_lines.append("Imports:")
        summary_lines.extend(f"- {imp}" for imp in summary_info["imports"])
    if summary_info.get("symbols"):
        summary_lines.append("Symbols:")
        summary_lines.extend(f"- {sym}" for sym in summary_info["symbols"])

    summary_text = "\n".join(summary_lines)
    return Chunk(
        repo_id=repo_id,
        file_path=rel_path,
        chunk_id=-1,
        text=summary_text,
        start_line=None,
        end_line=None,
        chunk_type="file_summary",
        dir_path=dir_path,
        language=language,
        embedding_text=_format_embedding_text(
            path=rel_path,
            dir_path=dir_path or ".",
            language=language,
            chunk_type="file_summary",
            start_line=None,
            end_line=None,
            body=summary_text,
        ),
    )


def _build_directory_summary_chunk(
    repo_id: str,
    dir_path: str,
    aggregate: dict,
) -> Chunk:
    files = aggregate.get("files", [])
    imports = list(aggregate.get("imports", []))
    symbols = list(aggregate.get("symbols", []))
    descriptors = []
    lowered = dir_path.lower()
    for keyword, desc in {
        "auth": "authentication logic",
        "security": "security features",
        "route": "API routing",
        "controller": "controllers",
        "service": "business services",
        "config": "configuration files",
        "util": "utilities/helpers",
        "model": "data models",
    }.items():
        if keyword in lowered:
            descriptors.append(desc)
    summary_lines = [
        f"Directory: {dir_path or '.'}",
        "Contains files:",
        *[f"- {name}" for name in files[:25]],
    ]
    if descriptors:
        summary_lines.append(f"Likely focus: {', '.join(descriptors)}")
    if imports:
        summary_lines.append("Common imports:")
        summary_lines.extend(f"- {imp}" for imp in imports[:20])
    if symbols:
        summary_lines.append("Key symbols:")
        summary_lines.extend(f"- {sym}" for sym in symbols[:25])
    summary_text = "\n".join(summary_lines)
    return Chunk(
        repo_id=repo_id,
        file_path=f"{dir_path or '.'}/",
        chunk_id=-2,
        text=summary_text,
        start_line=None,
        end_line=None,
        chunk_type="directory_summary",
        dir_path=dir_path,
        language="mixed",
        embedding_text=_format_embedding_text(
            path=f"{dir_path or '.'}/",
            dir_path=dir_path or ".",
            language="mixed",
            chunk_type="directory_summary",
            start_line=None,
            end_line=None,
            body=summary_text,
        ),
    )


def ingest_repository(
    repo_path: Path,
    output_path: Path,
    *,
    repo_id: str | None = None,
    ignore_dirs: Sequence[str] | None = None,
    allow_extensions: Iterable[str] | None = None,
    max_lines: int = 120,
    overlap: int = 30,
    encoding_fallbacks: Sequence[str] = DEFAULT_ENCODING_FALLBACKS,
    max_file_bytes: int = 750_000,
) -> list[Chunk]:
    """Traverse repository, chunk files, and write chunks to output_path."""

    repo_id = repo_id or repo_path.name
    repo_root = repo_path.resolve()

    chunks: list[Chunk] = []
    file_count = 0
    directory_aggregates: dict[str, dict] = defaultdict(
        lambda: {"files": [], "imports": [], "symbols": []}
    )

    for file in iter_repo_files(
        repo_root,
        ignore_dirs=ignore_dirs or DEFAULT_IGNORE_DIRS,
        allow_extensions=allow_extensions,
    ):
        relative_path = file.relative_to(repo_root).as_posix()
        if file.stat().st_size > max_file_bytes:
            continue
        if is_probably_binary(file):
            continue
        file_count += 1
        file_chunks, summary_info = chunk_file(
            repo_id=repo_id,
            file_path=file,
            rel_path=relative_path,
            max_lines=max_lines,
            overlap=overlap,
            encoding_fallbacks=encoding_fallbacks,
        )
        chunks.extend(file_chunks)
        file_summary_chunk = _build_file_summary_chunk(
            repo_id=repo_id,
            rel_path=relative_path,
            dir_path=summary_info.get("dir_path", ""),
            language=summary_info.get("language", "unknown"),
            summary_info=summary_info,
        )
        chunks.append(file_summary_chunk)
        dir_key = summary_info.get("dir_path", "")
        directory_aggregates[dir_key]["files"].append(Path(relative_path).name)
        directory_aggregates[dir_key]["imports"].extend(summary_info.get("imports", []))
        directory_aggregates[dir_key]["symbols"].extend(summary_info.get("symbols", []))

    for dir_path, aggregate in directory_aggregates.items():
        chunks.append(
            _build_directory_summary_chunk(
                repo_id=repo_id,
                dir_path=dir_path,
                aggregate=aggregate,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in chunks], f, ensure_ascii=False, indent=2)

    print(
        f"[F1] Repository ingestion complete: repo_id={repo_id} files={file_count} chunks={len(chunks)}"
    )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repository ingestion (F1)")
    parser.add_argument(
        "--repo-path",
        type=Path,
        help="Absolute or relative path to the repository to ingest",
    )
    parser.add_argument(
        "--repo-name",
        help="Name of a repo under ./repos (used if --repo-path is omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: embeddings/<repo_id>/chunks.json)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="Maximum lines per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Number of overlapping lines between consecutive chunks",
    )
    parser.add_argument(
        "--allow-extensions",
        help="Comma-separated list of file extensions to ingest (e.g. .py,.md)",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=750_000,
        help="Skip files larger than this many bytes",
    )
    return parser.parse_args()


def resolve_repo_path(repo_path: Path | None, repo_name: str | None) -> tuple[Path, str]:
    if repo_path and repo_name:
        raise ValueError("Provide either --repo-path or --repo-name, not both.")
    if repo_path:
        resolved = repo_path.resolve()
        return resolved, resolved.name
    if repo_name:
        resolved = (Path("repos") / repo_name).resolve()
        return resolved, repo_name
    raise ValueError("Must supply --repo-path or --repo-name.")


def main() -> None:
    args = parse_args()
    repo_root, repo_id = resolve_repo_path(args.repo_path, args.repo_name)
    output = args.output or Path("embeddings") / repo_id / "chunks.json"
    allow_exts = (
        [ext.strip() for ext in args.allow_extensions.split(",") if ext.strip()]
        if args.allow_extensions
        else None
    )

    ingest_repository(
        repo_root,
        output,
        repo_id=repo_id,
        allow_extensions=allow_exts,
        max_lines=args.max_lines,
        overlap=args.overlap,
        max_file_bytes=args.max_file_bytes,
    )


if __name__ == "__main__":
    main()
