"""JSONL tracing helpers for RAG answer requests."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRACE_LOCK = threading.Lock()


def resolve_trace_path(env: dict[str, str] | None = None) -> Path | None:
    raw = (env or os.environ).get("RAG_TRACE_PATH", "").strip()
    if not raw:
        return None
    return Path(raw)


def build_answer_trace(
    *,
    route: str,
    query: str,
    top_k: int,
    raw_query: bool,
    answer_mode: str,
    validation: dict[str, Any],
    retrieval_confidence: dict[str, Any],
    abstained: bool,
    verification: dict[str, Any],
    latency_ms: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "route": route,
        "query": query,
        "top_k": top_k,
        "raw_query": raw_query,
        "answer_mode": answer_mode,
        "validation_decision": validation.get("decision"),
        "confidence_level": retrieval_confidence.get("level"),
        "confidence_score": retrieval_confidence.get("score"),
        "abstained": abstained,
        "unsupported_count": len(verification.get("unsupported_sentences") or []),
        "latency_ms": latency_ms,
        "result_ids": [str(result.get("id", "")) for result in results],
        "result_titles": [str(result.get("title", "")) for result in results],
    }


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True)
    with _TRACE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL trace row: {exc}") from exc
        if isinstance(record, dict):
            rows.append(record)
    return rows


def summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "abstention_rate": 0.0,
            "warn_rate": 0.0,
            "average_latency_ms": 0.0,
            "answer_modes": {},
            "top_queries": [],
            "top_result_ids": [],
        }

    answer_modes = Counter(str(row.get("answer_mode", "")) for row in rows)
    validation_decisions = Counter(str(row.get("validation_decision", "")) for row in rows)
    queries = Counter(str(row.get("query", "")) for row in rows)
    result_ids: Counter[str] = Counter()
    for row in rows:
        for result_id in row.get("result_ids") or []:
            result_ids[str(result_id)] += 1

    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    count = len(rows)
    return {
        "count": count,
        "abstention_rate": round(sum(1 for row in rows if row.get("abstained")) / count, 4),
        "warn_rate": round(validation_decisions.get("warn", 0) / count, 4),
        "average_latency_ms": round(sum(latencies) / count, 2),
        "answer_modes": dict(answer_modes),
        "top_queries": queries.most_common(10),
        "top_result_ids": result_ids.most_common(10),
    }
