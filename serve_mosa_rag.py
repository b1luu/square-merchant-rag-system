#!/usr/bin/env python3
"""Warm local API for Mosa RAG: keep corpus, encoder, and FAISS resident in memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from mosa_rag.llm import DEFAULT_OLLAMA_MODEL, OLLAMA_LOCAL, OLLAMA_REMOTE, call_llm, stream_llm
from mosa_rag.runtime import ANSWER_TOP_K_DEFAULT, ResidentRetriever
from mosa_rag.validation_agent import validate_answer
from retrieve_pdf import MODEL_NAME

MAX_QUERY_CHARS = 1000
MAX_TOP_K = 20


def _sync_llm_env_from_args(args: argparse.Namespace) -> None:
    if args.ollama_provider.strip():
        os.environ["OLLAMA_PROVIDER"] = args.ollama_provider.strip()
    os.environ["OLLAMA_MODEL"] = (args.ollama_model or DEFAULT_OLLAMA_MODEL).strip()


def _validate_ollama_provider(name: str) -> None:
    if name not in (OLLAMA_LOCAL, OLLAMA_REMOTE):
        raise RuntimeError(
            f"OLLAMA_PROVIDER / --ollama-provider must be {OLLAMA_LOCAL!r} or {OLLAMA_REMOTE!r}, got {name!r}"
        )


@dataclass(frozen=True)
class RequestOptions:
    query: str
    top_k: int
    raw_query: bool
    show_context: bool
    stream: bool
    allow_low_confidence: bool


def _parse_bool_field(payload: dict, field: str, default: bool = False) -> bool:
    if field not in payload:
        return default
    value = payload[field]
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _parse_top_k(payload: dict, *, default: int) -> int:
    raw = payload.get("top_k", default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")
    if raw < 1 or raw > MAX_TOP_K:
        raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")
    return raw


def parse_request_options(
    payload: dict,
    *,
    top_k_default: int,
    raw_query_default: bool,
) -> RequestOptions:
    raw_query_text = payload.get("query", "")
    if not isinstance(raw_query_text, str):
        raise ValueError("query must be a string")
    query = raw_query_text.strip()
    if not query:
        raise ValueError("request JSON must include non-empty 'query'")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be {MAX_QUERY_CHARS} characters or fewer")

    return RequestOptions(
        query=query,
        top_k=_parse_top_k(payload, default=top_k_default),
        raw_query=_parse_bool_field(payload, "raw_query", raw_query_default),
        show_context=_parse_bool_field(payload, "show_context"),
        stream=_parse_bool_field(payload, "stream"),
        allow_low_confidence=_parse_bool_field(payload, "allow_low_confidence"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a warm local API for Mosa JSONL retrieval and answering.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--jsonl", type=Path, default=Path("normalized_mosa_rag.jsonl"))
    parser.add_argument("--model-name", default=MODEL_NAME, help=f"Embedding model (default: {MODEL_NAME})")
    parser.add_argument(
        "--top-k-default",
        type=int,
        default=ANSWER_TOP_K_DEFAULT,
        help=f"Default top-k for answer/retrieve requests (default: {ANSWER_TOP_K_DEFAULT})",
    )
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

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

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

        try:
            options = parse_request_options(
                payload,
                top_k_default=self.server.top_k_default,
                raw_query_default=self.server.raw_query_default,
            )
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if path == "/retrieve":
            self._handle_retrieve(query=options.query, top_k=options.top_k, raw_query=options.raw_query)
            return
        if path == "/answer":
            self._handle_answer(
                query=options.query,
                top_k=options.top_k,
                raw_query=options.raw_query,
                show_context=options.show_context,
                stream=options.stream,
                allow_low_confidence=options.allow_low_confidence,
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"unknown route: {path}"})

    def _handle_retrieve(self, *, query: str, top_k: int, raw_query: bool) -> None:
        started = time.time()
        results, confidence = self.server.retriever.retrieve_with_confidence(
            query,
            top_k=top_k,
            raw_query=raw_query,
        )
        hits = [asdict(hit) for hit in self.server.retriever.serialize_results(results)]
        self._write_json(
            HTTPStatus.OK,
            {
                "query": query,
                "top_k": top_k,
                "raw_query": raw_query,
                "latency_ms": round((time.time() - started) * 1000, 2),
                "retrieval_confidence": asdict(confidence),
                "results": hits,
            },
        )

    def _handle_answer(
        self,
        *,
        query: str,
        top_k: int,
        raw_query: bool,
        show_context: bool,
        stream: bool,
        allow_low_confidence: bool,
    ) -> None:
        if stream:
            self._handle_answer_stream(
                query=query,
                top_k=top_k,
                raw_query=raw_query,
                show_context=show_context,
                allow_low_confidence=allow_low_confidence,
            )
            return

        started = time.time()
        try:
            prompt, prompt_context, display_context, results, confidence = self.server.retriever.build_prompt_bundle(
                query,
                top_k=top_k,
                raw_query=raw_query,
            )
        except RuntimeError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        if confidence.should_abstain and not allow_low_confidence:
            verification = self.server.retriever.verify_answer("", [])
            payload = {
                "query": query,
                "top_k": top_k,
                "raw_query": raw_query,
                "latency_ms": round((time.time() - started) * 1000, 2),
                "abstained": True,
                "answer_mode": "abstain",
                "answer": "The retrieved records do not clearly answer this question.",
                "verification": asdict(verification),
                "validation": asdict(
                    validate_answer(
                        confidence=confidence,
                        verification=verification,
                        answer_mode="abstain",
                        abstained=True,
                    )
                ),
                "retrieval_confidence": asdict(confidence),
                "results": [asdict(hit) for hit in self.server.retriever.serialize_results(results)],
            }
            if show_context:
                payload["context"] = prompt_context
                payload["display_context"] = display_context
            self._write_json(HTTPStatus.OK, payload)
            return

        if not self.server.ollama_provider:
            answer = self.server.retriever.build_extractive_answer(results)
            verification = self.server.retriever.verify_answer(answer, results)
            payload = {
                "query": query,
                "top_k": top_k,
                "raw_query": raw_query,
                "latency_ms": round((time.time() - started) * 1000, 2),
                "abstained": False,
                "answer_mode": "extractive",
                "answer": answer,
                "verification": asdict(verification),
                "validation": asdict(
                    validate_answer(
                        confidence=confidence,
                        verification=verification,
                        answer_mode="extractive",
                        abstained=False,
                    )
                ),
                "retrieval_confidence": asdict(confidence),
                "results": [asdict(hit) for hit in self.server.retriever.serialize_results(results)],
            }
            if show_context:
                payload["context"] = prompt_context
                payload["display_context"] = display_context
            self._write_json(HTTPStatus.OK, payload)
            return

        try:
            answer = call_llm(prompt)
        except RuntimeError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        verification = self.server.retriever.verify_answer(answer, results)
        payload = {
            "query": query,
            "top_k": top_k,
            "raw_query": raw_query,
            "latency_ms": round((time.time() - started) * 1000, 2),
            "abstained": False,
            "answer_mode": "llm",
            "answer": answer,
            "verification": asdict(verification),
            "validation": asdict(
                validate_answer(
                    confidence=confidence,
                    verification=verification,
                    answer_mode="llm",
                    abstained=False,
                )
            ),
            "retrieval_confidence": asdict(confidence),
            "results": [asdict(hit) for hit in self.server.retriever.serialize_results(results)],
        }
        if show_context:
            payload["context"] = prompt_context
            payload["display_context"] = display_context
        self._write_json(HTTPStatus.OK, payload)

    def _handle_answer_stream(
        self,
        *,
        query: str,
        top_k: int,
        raw_query: bool,
        show_context: bool,
        allow_low_confidence: bool,
    ) -> None:
        started = time.time()
        try:
            prompt, prompt_context, display_context, results, confidence = self.server.retriever.build_prompt_bundle(
                query,
                top_k=top_k,
                raw_query=raw_query,
            )
        except RuntimeError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        if not self.server.ollama_provider and not (confidence.should_abstain and not allow_low_confidence):
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

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            titles = ", ".join(
                self.server.retriever.rows[chunk.chunk_id].get("title", "")
                for _, chunk in results
            )
            prelude_lines = [
                f"Retrieved {len(results)} record(s).",
                f"Confidence: level={confidence.level} score={confidence.score:.4f}",
                f"Retrieved: {titles}",
                "",
            ]
            self.wfile.write(("\n".join(prelude_lines)).encode("utf-8"))
            self.wfile.flush()
            if confidence.should_abstain and not allow_low_confidence:
                reason_text = ""
                if confidence.reasons:
                    reason_text = f"Reason: {'; '.join(confidence.reasons)}\n"
                self.wfile.write(
                    (
                        "The retrieved records do not clearly answer this question.\n"
                        f"{reason_text}"
                    ).encode("utf-8")
                )
                self.wfile.flush()
                return
            self.wfile.write(f"Generating answer from {len(results)} retrieved record(s)...\n\n".encode("utf-8"))
            self.wfile.flush()
            for piece in stream_llm(prompt):
                self.wfile.write(piece.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except RuntimeError as exc:
            try:
                self.wfile.write(f"\n\n[stream error] {exc}\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        trailer = [f"\n\nLatency: {round((time.time() - started) * 1000, 2)} ms"]
        if show_context:
            trailer.append(f"Prompt Context\n{prompt_context}")
            trailer.append(f"Display Context\n{display_context}")

        try:
            self.wfile.write(("\n\n" + "\n\n".join(trailer) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

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
