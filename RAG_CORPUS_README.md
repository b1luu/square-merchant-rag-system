# Normalized JSONL corpus for RAG

This repository can hold a **normalized JSONL knowledge base** built from your own PDFs. Records are structured for **dense retrieval** (e.g. BGE + FAISS) by storing **atomic operational units** instead of raw page-level text.

## Why this structure (vs. raw PDF chunking)

- **Atomic concepts**: Each JSONL line is one main idea (e.g. one recipe, one procedure, one policy topic). Embeddings align better with short, natural staff questions.
- **Structured fields**: Quantities, times, ordered steps, and rules can live in typed fields (`ingredients`, `steps`, `rules`, `threshold`, `storage_life`, etc.) and in natural language inside `retrieval_text`.
- **Cleaner embedding text**: `retrieval_text` can drop PDF line-wrap noise and add light context so informal queries still match.
- **Traceability**: `source_page` and paths are metadata for auditing, not the primary chunking strategy.

## Source PDFs

Put inputs under `data/` (or another directory you pass to the builder). Which files are ingested and how they map to extractors is **project-specific**—see `mosa_rag/build_corpus.py` and the `mosa_rag/extract_*.py` modules.

## Build

From the repo root (with dependencies installed):

```bash
python -m mosa_rag.build_corpus --data-dir data --out normalized_mosa_rag.jsonl
```

Adjust `--out` if you use a different corpus filename.

## Validate

```bash
python validate_mosa_rag_jsonl.py normalized_mosa_rag.jsonl
```

## Semantic search (BGE + FAISS, no LLM)

After the JSONL exists:

```bash
python retrieve_mosa_rag_jsonl.py "How do I cook boba in the rice cooker?"
python retrieve_mosa_rag_jsonl.py "closing checklist for the register" --top-k 5
```

Use `--raw-query` to omit the BGE query-instruction prefix (for A/B comparison). The embedding model may download on first use.

### Index cache

Re-embedding the full corpus every run is optional. By default, an on-disk cache lives next to the JSONL under `.rag_index_cache/` (or set **`BGE_RAG_CACHE_DIR`**). Use **`--no-cache`** to force a full rebuild.

## Retrieval evaluation

Curated query sets under `eval_sets/` (filenames may vary by checkout):

```bash
python evaluate_mosa_rag_jsonl.py
python evaluate_mosa_rag_jsonl.py eval_sets/mosa_rag_gap_probes.jsonl --top-k 3
```

## LLM answers on top of retrieval

`answer_mosa_rag_jsonl.py` runs the same retrieval path, then calls a local **Ollama** model via HTTP (`/api/generate`). No cloud API key is required for the default setup.

Inspect the composed prompt without calling the model:

```bash
export OLLAMA_PROVIDER=ollama_local
python answer_mosa_rag_jsonl.py "What is the sick leave policy?" --dry-run
```

Run a live answer (Ollama must be running; pull your model first, e.g. `ollama pull llama3.2`):

```bash
export OLLAMA_PROVIDER=ollama_local
python answer_mosa_rag_jsonl.py "What is the sick leave policy?" --show-context
```

Remote or self-hosted Ollama (same HTTP API, e.g. another machine or a container):

```bash
export OLLAMA_PROVIDER=ollama_remote
export OLLAMA_BASE_URL=https://your-ollama-host.example.com
export OLLAMA_MODEL=llama3.2
python answer_mosa_rag_jsonl.py "Your question"
```

Useful environment variables:

| Variable | Role |
|----------|------|
| `OLLAMA_PROVIDER` | `ollama_local` (default base `http://localhost:11434`) or `ollama_remote` (`OLLAMA_BASE_URL`) |
| `OLLAMA_BASE_URL` | Base URL for `ollama_remote` (no `/api/generate` suffix) |
| `OLLAMA_MODEL` | Model tag (default in code: `llama3.2`) |
| `OLLAMA_TEMPERATURE` | Optional; default `0` in the client for steadier answers |

CLI flags include **`--ollama-model`**, **`--ollama-provider`**, **`--top-k`**, **`--show-context`**, **`--dry-run`**, **`--cache-dir`**, **`--no-cache`**.

## Case format (eval JSONL)

- `id` — stable case identifier  
- `category` — grouping for summary metrics  
- `query` — natural-language search query  
- `expected_titles` — record titles that count as correct  
- `expected_ids` — optional; brittle if IDs shift when records are inserted  
- `notes` — optional context for humans  

## Record integrity (validator)

- Required string fields: `id`, `type`, `title`, `retrieval_text`  
- `id` uniqueness  
- `type` must be an allowed record type  
- `retrieval_text` non-empty  

## Code layout

- `mosa_rag/pdf_text.py` — PDF text extraction  
- `mosa_rag/normalize.py` — whitespace cleanup  
- `mosa_rag/extract_*.py` — document-family extractors  
- `mosa_rag/build_corpus.py` — assigns IDs and writes JSONL  
- `mosa_rag/retrieve_jsonl.py` — load rows and build retrieval chunks  
- `mosa_rag/faiss_cache.py` — optional persisted FAISS index  
- `mosa_rag/llm.py` — Ollama HTTP adapter (`call_llm`)  
- `validate_mosa_rag_jsonl.py` — lightweight checks  

## Extending

1. Add or extend an extractor under `mosa_rag/`.  
2. Register it in `mosa_rag/build_corpus.py::build_records`.  
3. For a new `type`, add it to `ALLOWED_TYPES` in `mosa_rag/schema.py` and update the validator if needed.  

## Notes

- **No OCR in the default path**: text comes from PDF extraction. Scan-only PDFs need OCR upstream.  
- **External links** in source PDFs are not automatically normalized into the JSONL unless your extractor adds them.  
