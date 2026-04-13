#!/usr/bin/env python3
"""Warm local API for Mosa RAG: keep corpus, encoder, and FAISS resident in memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from mosa_rag.llm import DEFAULT_OLLAMA_MODEL, OLLAMA_LOCAL, OLLAMA_REMOTE, call_llm
from mosa_rag.runtime import ResidentRetriever
from retrieve_pdf import MODEL_NAME


def _sync_llm_env_from_args(args: argparse.Namespace) -> None:
    if args.ollama_provider.strip():
        os.environ["OLLAMA_PROVIDER"] = args.ollama_provider.strip()
    os.environ["OLLAMA_MODEL"] = (args.ollama_model or DEFAULT_OLLAMA_MODEL).strip()


def _validate_ollama_provider(name: str) -> None:
    if name not in (OLLAMA_LOCAL, OLLAMA_REMOTE):
        raise RuntimeError(
            f"OLLAMA_PROVIDER / --ollama-provider must be {OLLAMA_LOCAL!r} or {OLLAMA_REMOTE!r}, got {name!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a warm local API for Mosa JSONL retrieval and answering.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--jsonl", type=Path, default=Path("normalized_mosa_rag.jsonl"))
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"Embedding model (default: {MODEL_NAME})")
    parser.add_argument("--top-k-default", type=int, default=5, help="Default top-k for requests (default: 5)")
    parser.add_argument("--raw-query-default", action="store_true", help="Disable BGE query prefix by default")
    parser.add_argument(
        "--ollama-provider",
        default=os.getenv("OLLAMA_PROVIDER", ""),
        metavar="NAME",
        help=f"{OLLAMA_LOCAL} or {OLLAMA_REMOTE}; default from OLLAMA_PROVIDER env",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=f"Ollama model tag (default: {DEFAULT_OLLAMA_MODEL} or OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for FAISS cache (default: BGE_RAG_CACHE_DIR env, else .rag_index_cache next to --jsonl)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Always rebuild embeddings and FAISS index")
    return parser.parse_args()


class MosaRagServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        retriever: ResidentRetriever,
        top_k_default: int,
        raw_query_default: bool,
        ollama_provider: str,
        ollama_model: str,
        started_at: float,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.retriever = retriever
        self.top_k_default = top_k_default
        self.raw_query_default = raw_query_default
        self.ollama_provider = ollama_provider
        self.ollama_model = ollama_model
        self.started_at = started_at


class RequestHandler(BaseHTTPRequestHandler):
    server: MosaRagServer

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            payload = {
                "status": "ok",
                "records": len(self.server.retriever.rows),
                "jsonl": str(self.server.retriever.jsonl),
                "embedding_model": self.server.retriever.model_name,
                "ollama_provider": self.server.ollama_provider or None,
                "ollama_model": self.server.ollama_model,
                "uptime_seconds": round(time.time() - self.server.started_at, 3),
            }
            self._write_json(HTTPStatus.OK, payload)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"unknown route: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        query = str(payload.get("query", "")).strip()
        if not query:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "request JSON must include non-empty 'query'"})
            return

        top_k = int(payload.get("top_k") or self.server.top_k_default)
        raw_query = bool(payload.get("raw_query", self.server.raw_query_default))

        if path == "/retrieve":
            self._handle_retrieve(query=query, top_k=top_k, raw_query=raw_query)
            return
        if path == "/answer":
            self._handle_answer(query=query, top_k=top_k, raw_query=raw_query, show_context=bool(payload.get("show_context")))
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"unknown route: {path}"})

    def _handle_retrieve(self, *, query: str, top_k: int, raw_query: bool) -> None:
        started = time.time()
        results = self.server.retriever.retrieve(query, top_k=top_k, raw_query=raw_query)
        hits = [asdict(hit) for hit in self.server.retriever.serialize_results(results)]
        self._write_json(
            HTTPStatus.OK,
            {
                "query": query,
                "top_k": top_k,
                "raw_query": raw_query,
                "latency_ms": round((time.time() - started) * 1000, 2),
                "results": hits,
            },
        )

    def _handle_answer(self, *, query: str, top_k: int, raw_query: bool, show_context: bool) -> None:
        if not self.server.ollama_provider:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": (
                        f"server has no Ollama provider configured; start it with --ollama-provider "
                        f"{OLLAMA_LOCAL} or {OLLAMA_REMOTE}"
                    )
                },
            )
            return

        started = time.time()
        try:
            prompt, context, results = self.server.retriever.build_prompt_bundle(
                query,
                top_k=top_k,
                raw_query=raw_query,
            )
            answer = call_llm(prompt)
        except RuntimeError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        payload = {
            "query": query,
            "top_k": top_k,
            "raw_query": raw_query,
            "latency_ms": round((time.time() - started) * 1000, 2),
            "answer": answer,
            "results": [asdict(hit) for hit in self.server.retriever.serialize_results(results)],
        }
        if show_context:
            payload["context"] = context
        self._write_json(HTTPStatus.OK, payload)

    def _read_json(self) -> dict:
        raw_length = self.headers.get("Content-Length", "").strip()
        if not raw_length:
            raise ValueError("missing Content-Length")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError(f"invalid Content-Length: {raw_length!r}") from exc
        raw_body = self.rfile.read(length)
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("request JSON must be an object")
        return data

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        sys.exit(f"error: corpus file not found: {args.jsonl}")
    if args.ollama_provider.strip():
        try:
            _validate_ollama_provider(args.ollama_provider.strip())
        except RuntimeError as exc:
            sys.exit(f"error: {exc}")
    _sync_llm_env_from_args(args)

    try:
        retriever = ResidentRetriever(
            jsonl=args.jsonl,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    started_at = time.time()
    server = MosaRagServer(
        (args.host, args.port),
        RequestHandler,
        retriever=retriever,
        top_k_default=args.top_k_default,
        raw_query_default=args.raw_query_default,
        ollama_provider=args.ollama_provider.strip(),
        ollama_model=args.ollama_model,
        started_at=started_at,
    )
    print(
        f"Mosa RAG server ready on http://{args.host}:{args.port} "
        f"(records={len(retriever.rows)}, embedding_model={retriever.model_name})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
