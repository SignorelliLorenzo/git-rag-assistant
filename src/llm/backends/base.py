"""LLM backend abstractions for F5 answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for a single generation request."""

    temperature: float = 0.1
    max_tokens: int = 512
    stop: Sequence[str] | None = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    name: str

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate completion text for the provided prompt."""


def _normalize_backend_name(name: str) -> str:
    return name.strip().lower()


def load_backend(name: str, **kwargs) -> LLMBackend:
    """Factory for loading a backend by name."""

    normalized = _normalize_backend_name(name)
    if normalized in {"ollama", "", None}:
        from .ollama_backend import OllamaBackend

        return OllamaBackend(**kwargs)

    raise ValueError(
        f"Unsupported LLM backend '{name}'. Available backend: ollama."
    )
