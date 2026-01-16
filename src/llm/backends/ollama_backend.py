"""LLM backend powered by Ollama."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import ollama

from .base import GenerationConfig, LLMBackend

logger = logging.getLogger("git_rag")


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(
        self,
        model_path: str,
        ollama_host: str | None = None,
        request_id: str | None = None,
        **_: object,
    ) -> None:
        # model_path is the model name/tag in Ollama (e.g., "llama3:13b").
        self._model = model_path
        self._request_id = request_id
        self._client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()

    @staticmethod
    def _build_options(config: GenerationConfig) -> dict:
        options: dict[str, object] = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        }
        if config.stop:
            options["stop"] = list(config.stop)
        return options

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        options = self._build_options(config)
        logger.info(
            "ollama_call_start request_id=%s model=%s options=%s",
            self._request_id,
            self._model,
            options,
        )
        t0 = time.perf_counter()
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                stream=False,
                options=options,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "ollama_call_failed request_id=%s model=%s duration_ms=%.2f",
                self._request_id,
                self._model,
                (time.perf_counter() - t0) * 1000,
            )
            raise

        duration_ms = (time.perf_counter() - t0) * 1000
        text = str(response.get("response", "")).strip()
        logger.info(
            "ollama_call_done request_id=%s model=%s duration_ms=%.2f response_chars=%s",
            self._request_id,
            self._model,
            duration_ms,
            len(text),
        )
        return text
