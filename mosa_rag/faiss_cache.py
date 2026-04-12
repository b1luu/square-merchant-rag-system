"""Persist FAISS + BGE embeddings for a JSONL corpus to skip full re-encode on each run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from retrieve_pdf import BGEOnnxEncoder, Chunk, build_faiss_index

CACHE_SCHEMA_VERSION = 1
META_NAME = "meta.json"
INDEX_NAME = "index.faiss"


def default_cache_root(source_jsonl: Path) -> Path:
    return source_jsonl.resolve().parent / ".rag_index_cache"


def _bucket_name(source_jsonl: Path, model_name: str) -> str:
    key = f"{CACHE_SCHEMA_VERSION}\0{source_jsonl.resolve()}\0{model_name}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _embeddings_from_index(index: faiss.IndexFlatIP) -> np.ndarray:
    n = int(index.ntotal)
    if n == 0:
        return np.zeros((0, int(index.d)), dtype=np.float32)
    rows = [index.reconstruct(i) for i in range(n)]
    return np.vstack(rows).astype(np.float32)


def _try_load_cache(
    cache_dir: Path,
    source_jsonl: Path,
    model_name: str,
    num_chunks: int,
) -> tuple[BGEOnnxEncoder, faiss.IndexFlatIP, np.ndarray] | None:
    meta_path = cache_dir / META_NAME
    index_path = cache_dir / INDEX_NAME
    if not meta_path.is_file() or not index_path.is_file():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    try:
        st = source_jsonl.stat()
    except OSError:
        return None

    if (
        meta.get("schema") != CACHE_SCHEMA_VERSION
        or meta.get("jsonl") != str(source_jsonl.resolve())
        or meta.get("mtime_ns") != st.st_mtime_ns
        or meta.get("size_bytes") != st.st_size
        or meta.get("model_name") != model_name
        or meta.get("num_chunks") != num_chunks
    ):
        return None

    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError:
        return None

    if not isinstance(index, faiss.IndexFlatIP):
        return None
    if int(index.ntotal) != num_chunks:
        return None

    encoder = BGEOnnxEncoder(model_name)
    probe = encoder.encode(["__cache_probe__"])
    if int(probe.shape[1]) != int(index.d):
        return None

    embeddings = _embeddings_from_index(index)
    return encoder, index, embeddings


def _write_cache(
    cache_dir: Path,
    source_jsonl: Path,
    model_name: str,
    num_chunks: int,
    index: faiss.IndexFlatIP,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / INDEX_NAME
    meta_path = cache_dir / META_NAME
    st = source_jsonl.stat()
    meta = {
        "schema": CACHE_SCHEMA_VERSION,
        "jsonl": str(source_jsonl.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size_bytes": st.st_size,
        "model_name": model_name,
        "num_chunks": num_chunks,
        "index_name": INDEX_NAME,
    }
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_or_build_faiss_index(
    chunks: list[Chunk],
    model_name: str,
    source_jsonl: Path,
    *,
    cache_root: Path | None,
    no_cache: bool = False,
) -> tuple[BGEOnnxEncoder, faiss.IndexFlatIP, np.ndarray]:
    """
    Same contract as retrieve_pdf.build_faiss_index, but reuses a saved index when the
    JSONL path, size, mtime, embedding model, and chunk count match cache metadata.
    """
    if no_cache or cache_root is None:
        return build_faiss_index(chunks=chunks, model_name=model_name)

    bucket = cache_root / _bucket_name(source_jsonl, model_name)
    loaded = _try_load_cache(bucket, source_jsonl, model_name, len(chunks))
    if loaded is not None:
        print(
            f"RAG index: loaded {len(chunks)} vectors from cache ({bucket.name})",
            file=sys.stderr,
        )
        return loaded

    encoder, index, embeddings = build_faiss_index(chunks=chunks, model_name=model_name)
    try:
        _write_cache(bucket, source_jsonl, model_name, len(chunks), index)
        print(
            f"RAG index: built and cached {len(chunks)} vectors ({bucket.name})",
            file=sys.stderr,
        )
    except OSError:
        print("RAG index: built (cache write failed; next run will rebuild)", file=sys.stderr)
    return encoder, index, embeddings
