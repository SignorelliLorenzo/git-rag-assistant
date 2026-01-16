"""CLI entrypoint for F5 answer generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .answer_generation import generate_answer
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers grounded in repo context (F5)")
    parser.add_argument("--repo-id", required=True, help="Repository identifier.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument(
        "--hits",
        type=Path,
        required=True,
        help="Path to JSON file containing retrieval hits (list of dicts with 'text' fields).",
    )
    parser.add_argument(
        "--backend",
        default="llama-cpp",
        help="LLM backend to use (default: llama-cpp). Options: llama-cpp, ctransformers.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Backend-specific model identifier (e.g., GGUF path for llama-cpp).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Maximum characters of repository context to include in the prompt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of retrieval hits to include.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--stop",
        help="Optional comma-separated stop sequences.",
    )
    parser.add_argument(
        "--device",
        help="Preferred execution device for the LLM backend (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--model-type",
        help="Model type identifier (ctransformers backend, e.g., llama, mistral).",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        help="Override number of GPU layers/offloaded layers (backend-specific).",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        help="Override number of CPU threads (backend-specific).",
    )
    return parser.parse_args()


ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")


def _read_text_with_fallbacks(path: Path) -> str:
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise UnicodeDecodeError("unknown", raw, 0, 1, "Unable to decode retrieval hits file")


def _load_retrieval_hits(path: Path) -> list[dict]:
    text = _read_text_with_fallbacks(path).strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    hits: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hits.append(json.loads(line))
    return hits


def main() -> None:
    args = parse_args()
    if not args.hits.exists():
        raise FileNotFoundError(f"Retrieval hits file not found at {args.hits}")
    retrieval_hits = _load_retrieval_hits(args.hits)

    stop_sequences = (
        [seq.strip() for seq in args.stop.split(",") if seq.strip()] if args.stop else None
    )

    result = generate_answer(
        repo_id=args.repo_id,
        question=args.question,
        retrieval_hits=retrieval_hits,
        backend=args.backend,
        model_id=args.model,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=stop_sequences,
        device=args.device,
        model_type=args.model_type,
        gpu_layers=args.gpu_layers,
        n_threads=args.n_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
