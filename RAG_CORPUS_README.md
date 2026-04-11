# Mosa Tea — Normalized RAG Corpus

This folder contains a **normalized JSONL knowledge base** derived from Mosa’s SOP PDFs and employee handbook. It is designed for **semantic retrieval with BGE** (and similar dense embedding models) by storing **atomic operational units** instead of raw PDF text or page-based chunks.

## Why this structure (vs. raw PDF chunking)

- **Atomic concepts**: Each JSONL line is one main idea (one drink build, one batch recipe, one POS workflow, one policy topic). That improves precision/recall because embeddings align with how staff ask questions.
- **Operational facts preserved**: Quantities, temperatures, time windows, shelf life, and ordered steps live in structured fields (`ingredients`, `steps`, `rules`, `threshold`, `storage_life`, etc.) and are repeated in natural language inside `retrieval_text`.
- **Cleaner text for embeddings**: `retrieval_text` removes PDF line-wrap noise and states context (“bar staff”, “kitchen opening”, “Square POS”) so queries like “how do I clock in?” match even if the PDF wording is scattered.
- **No page-only chunking**: `source_page` is metadata for traceability, not a segmentation strategy.

## Source files (expected in `data/`)

| Role | File |
|------|------|
| Bar SOP | `SOP - Bar (1).pdf` |
| Kitchen opening | `SOP - Kitchen Open.pdf` |
| Kitchen batch / tea brewing | `SOP - Kitchen.pdf` |
| Employee handbook | `Mosa Employee Handbook.pdf` (if your copy is named differently, keep the same content or adjust `DEFAULT_SOURCES` in `mosa_rag/build_corpus.py`) |
| Towel color code | `Cleaning Towels Color Code.pdf` |

## Build

From the repo root (with dependencies installed):

```bash
python -m mosa_rag.build_corpus --data-dir data --out normalized_mosa_rag.jsonl
```

## Validate

```bash
python validate_mosa_rag_jsonl.py normalized_mosa_rag.jsonl
```

## Try semantic search (same BGE model as `retrieve_pdf.py`)

After `normalized_mosa_rag.jsonl` exists, run ad hoc queries against **`retrieval_text`** records (FAISS + ONNX BGE, no LLM):

```bash
python retrieve_mosa_rag_jsonl.py "How do I cook boba in the rice cooker?"
python retrieve_mosa_rag_jsonl.py "Square POS passcode" --top-k 5
```

Use `--raw-query` to omit the BGE query instruction prefix (for A/B comparison). First run may download the model; embedding 126 records is quick after that.

Checks:

- required string fields: `id`, `type`, `title`, `retrieval_text`
- `id` uniqueness
- `type` is one of the allowed record types
- `retrieval_text` is non-empty

## Code layout

- `mosa_rag/pdf_text.py` — pypdf per-page extraction
- `mosa_rag/normalize.py` — whitespace cleanup for parsing and canonical text
- `mosa_rag/extract_*.py` — one module per document family (bar, kitchen batch, kitchen open, handbook, towels)
- `mosa_rag/build_corpus.py` — assigns stable IDs and writes JSONL
- `validate_mosa_rag_jsonl.py` — lightweight integrity checks

## Extending

1. Add a new extractor module under `mosa_rag/` (or extend an existing one).
2. Register it in `mosa_rag/build_corpus.py::build_records`.
3. If you add a new `type`, add it to `ALLOWED_TYPES` in `mosa_rag/schema.py` and update the validator if needed.

## Notes

- **No OCR**: text comes from PDF extraction (`pypdf`). If a future PDF is scan-only, you’ll need an OCR step upstream.
- **Handbook links**: the handbook references external guidelines (e.g., county food safety). Those URLs are not embedded as live links in this corpus—add them if you want retrieval to surface the exact link text.
