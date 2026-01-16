"""High-level API for F5 answer generation."""

from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

from .backends.base import GenerationConfig, load_backend
from .prompt_builder import PromptBundle, build_prompt

logger = logging.getLogger("git_rag")


def _ensure_hits_iterable(retrieval_hits: Iterable[dict]) -> list[dict]:
    if isinstance(retrieval_hits, list):
        return retrieval_hits
    return list(retrieval_hits)


def generate_answer(
    *,
    repo_id: str,
    question: str,
    retrieval_hits: Iterable[dict],
    backend: str,
    model_id: str,
    ollama_host: str | None = None,
    top_k: int = 5,
    max_context_chars: int = 12_000,
    temperature: float = 0.1,
    max_tokens: int = 512,
    stop: Sequence[str] | None = None,
    device: str | None = None,
    model_type: str | None = None,
    gpu_layers: int | None = None,
    n_threads: int | None = None,
    request_id: str | None = None,
) -> dict:
    """Generate a grounded answer using retrieved chunks."""
    t0_total = time.perf_counter()
    hits = _ensure_hits_iterable(retrieval_hits)
    if not hits:
        prompt_bundle = PromptBundle(
            prompt="No repository context provided.\nAnswer: ",
            used_chunks=[],
            prompt_info={
                "top_k_requested": top_k,
                "hit_count_provided": 0,
                "context_chars": 0,
                "max_context_chars": max_context_chars,
                "context_truncated": False,
            },
        )
    else:
        t0 = time.perf_counter()
        prompt_bundle = build_prompt(
            question=question,
            hits=hits,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )
        logger.info(
            "prompt_built request_id=%s repo_id=%s hits_in=%s hits_used=%s context_chars=%s truncated=%s build_ms=%.2f",
            request_id,
            repo_id,
            len(hits),
            prompt_bundle.prompt_info.get("hit_count_provided"),
            prompt_bundle.prompt_info.get("context_chars"),
            prompt_bundle.prompt_info.get("context_truncated"),
            (time.perf_counter() - t0) * 1000,
        )

    backend_normalized = backend.strip().lower()
    backend_kwargs: dict[str, object] = {"model_path": model_id, "request_id": request_id}
    if backend_normalized == "ollama":
        backend_kwargs["ollama_host"] = ollama_host
    if backend_normalized == "llama-cpp":
        backend_kwargs.update(
            {
                "device": device,
                "n_threads": n_threads,
            }
        )
    if backend_normalized == "ctransformers":
        backend_kwargs.update(
            {
                "device": device,
                "model_type": model_type,
                "gpu_layers": gpu_layers,
                "n_threads": n_threads,
            }
        )

    backend_client = load_backend(backend, **backend_kwargs)
    generation_config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    logger.info(
        "llm_generate_start request_id=%s backend=%s model_id=%s temperature=%s max_tokens=%s has_stop=%s",
        request_id,
        backend_client.name,
        model_id,
        temperature,
        max_tokens,
        bool(stop),
    )
    t0 = time.perf_counter()
    answer = backend_client.generate(prompt_bundle.prompt, generation_config)
    gen_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "llm_generate_done request_id=%s backend=%s model_id=%s gen_ms=%.2f total_ms=%.2f",
        request_id,
        backend_client.name,
        model_id,
        gen_ms,
        (time.perf_counter() - t0_total) * 1000,
    )

    return {
        "repo_id": repo_id,
        "question": question,
        "answer": answer,
        "used_chunks": prompt_bundle.used_chunks,
        "model_info": {
            "backend": backend_client.name,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "prompt_info": prompt_bundle.prompt_info,
    }
