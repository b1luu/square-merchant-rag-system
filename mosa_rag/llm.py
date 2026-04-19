"""LLM adapter: Ollama HTTP `/api/generate` (local, remote VM, or Cloud Run with the same API)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from urllib import error, request

# Provider names (OLLAMA_PROVIDER). Same HTTP contract works for GCP-hosted Ollama
# (Compute Engine VM or Cloud Run) via OLLAMA_BASE_URL.
OLLAMA_LOCAL = "ollama_local"
OLLAMA_REMOTE = "ollama_remote"

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_LOCAL_BASE = "http://localhost:11434"


def _effective_provider() -> str:
    return os.environ.get("OLLAMA_PROVIDER", "").strip()


def _effective_model() -> str:
    m = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    return m or DEFAULT_OLLAMA_MODEL


def resolve_ollama_generate_url() -> str:
    """Return the full `/api/generate` URL for the current OLLAMA_PROVIDER."""
    provider = _effective_provider()
    if provider == OLLAMA_LOCAL:
        base = DEFAULT_LOCAL_BASE.rstrip("/")
    elif provider == OLLAMA_REMOTE:
        base = os.environ.get("OLLAMA_BASE_URL", "").strip().rstrip("/")
        if not base:
            raise RuntimeError("OLLAMA_BASE_URL must be set when OLLAMA_PROVIDER=ollama_remote")
    else:
        raise RuntimeError(
            f"OLLAMA_PROVIDER must be {OLLAMA_LOCAL!r} or {OLLAMA_REMOTE!r}, got {provider!r}"
        )
    return f"{base}/api/generate"


def _ollama_options() -> dict[str, float]:
    """Ollama /api/generate `options`; defaults favor stable RAG answers."""
    raw = os.getenv("OLLAMA_TEMPERATURE")
    if raw is not None and raw.strip() != "":
        return {"temperature": float(raw)}
    return {"temperature": 0.0}


def _build_ollama_request(prompt: str, *, stream: bool) -> request.Request:
    body: dict = {
        "model": _effective_model(),
        "prompt": prompt,
        "stream": stream,
        "options": _ollama_options(),
    }
    payload = json.dumps(body).encode("utf-8")
    return request.Request(
        resolve_ollama_generate_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def call_llm(prompt: str) -> str:
    """
    Send `prompt` to Ollama and return the model's plain-text reply.

    Configuration (environment):
      OLLAMA_PROVIDER — 'ollama_local' (http://localhost:11434) or 'ollama_remote' (OLLAMA_BASE_URL)
      OLLAMA_BASE_URL — required for ollama_remote (e.g. https://ollama.example.com or a Cloud Run URL)
      OLLAMA_MODEL — model tag (default: llama3.2)
      OLLAMA_TEMPERATURE — sampling temperature (default: 0.0 for repeatable answers; raise for variety)
    """
    req = _build_ollama_request(prompt, stream=False)
    try:
        with request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error ({exc.code}): {err_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    text = body.get("response")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"unexpected Ollama response (no 'response' text): {body!r}")
    return text.strip()


def stream_llm(prompt: str) -> Iterator[str]:
    """Yield plain-text response chunks from Ollama as they arrive."""
    req = _build_ollama_request(prompt, stream=True)
    try:
        resp = request.urlopen(req, timeout=120)
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error ({exc.code}): {err_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def _iter_chunks() -> Iterator[str]:
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    body = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"unexpected Ollama stream payload: {line!r}") from exc
                text = body.get("response")
                if isinstance(text, str) and text:
                    yield text

    return _iter_chunks()
