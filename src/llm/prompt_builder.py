"""Utilities to build prompts for F5 answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


SYSTEM_PROMPT = (
    "You are GitRepo RAG Assistant. Answer questions about the repository using only "
    "the provided SOURCE snippets. Cite sources by referencing their SOURCE labels. "
    "If the information is not present in the snippets, say you do not know."
)


@dataclass
class PromptBundle:
    prompt: str
    used_chunks: List[dict]
    prompt_info: dict = field(default_factory=dict)


def _normalize_hits(hits: Iterable[dict]) -> List[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str | None, int | None]] = set()

    for hit in hits:
        text = str(hit.get("text", "")).strip()
        if not text:
            continue
        key = (hit.get("file_path"), hit.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(hit)
    return normalized


def _format_source_block(source_id: str, chunk: dict) -> str:
    file_path = chunk.get("file_path", "unknown file")
    start = chunk.get("start_line")
    end = chunk.get("end_line")
    line_label = (
        f"(lines {start}-{end})"
        if isinstance(start, int) and isinstance(end, int)
        else "(lines n/a)"
    )
    chunk_id = chunk.get("chunk_id", "n/a")
    header = f"{source_id}: {file_path} {line_label} [chunk_id={chunk_id}]"
    text = str(chunk.get("text", "")).strip()
    return f"{header}\n{text}\n"


def build_prompt(
    question: str,
    hits: Iterable[dict],
    *,
    top_k: int = 5,
    max_context_chars: int = 12000,
    system_prompt: str = SYSTEM_PROMPT,
) -> PromptBundle:
    normalized_hits = _normalize_hits(hits)
    selected_hits = normalized_hits[: max(top_k, 1)]

    context_blocks: list[str] = []
    used_chunks: list[dict] = []
    total_chars = 0
    truncated = False

    for idx, chunk in enumerate(selected_hits, start=1):
        source_id = f"SOURCE {idx}"
        block = _format_source_block(source_id, chunk)
        if total_chars + len(block) > max_context_chars:
            truncated = True
            break
        context_blocks.append(block)
        enriched_chunk = dict(chunk)
        enriched_chunk["source_id"] = source_id
        used_chunks.append(enriched_chunk)
        total_chars += len(block)

    if not context_blocks:
        context_text = "No repository context was retrieved for this question."
    else:
        context_text = "\n".join(context_blocks).strip()

    question_clean = question.strip()
    prompt = (
        f"{system_prompt}\n\n"
        f"Question:\n{question_clean}\n\n"
        f"Repository Context:\n{context_text}\n\n"
        "Instructions:\n"
        "- Provide a concise answer grounded in the context above.\n"
        "- Cite sources inline using the SOURCE labels.\n"
        "- If the answer cannot be found in the context, say so explicitly.\n\n"
        "Answer:\n"
    )

    prompt_info = {
        "top_k_requested": top_k,
        "hit_count_provided": len(used_chunks),
        "context_chars": total_chars,
        "max_context_chars": max_context_chars,
        "context_truncated": truncated,
    }
    return PromptBundle(prompt=prompt, used_chunks=used_chunks, prompt_info=prompt_info)
