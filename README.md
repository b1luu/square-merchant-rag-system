# Mosa Ops RAG System

## Overview

**Mosa Ops RAG System** is a retrieval-augmented assistant for **real shift operations**: bar and back-of-house staff get **grounded answers** over SOPs, recipes, and policy text instead of paging PDFs or guessing. The system **engineers retrieval first**—dense vectors over a curated **JSONL** knowledge base—then optionally applies an **LLM** only to **interpret and summarize** the top-k hits, so answers stay tied to source material. It is designed to **replace slow manual lookup** (flip, search, ask a manager) with a **single query path** that returns cited, staff-readable output in **seconds to well under a minute** on typical laptop hardware.

---

## Quick Demo

With **Ollama** running locally and `OLLAMA_PROVIDER=ollama_local`:

```bash
export OLLAMA_PROVIDER=ollama_local
python answer_mosa_rag_jsonl.py "What happens if I am sick?"
```

**Output (illustrative; exact text depends on retrieved records and model):**

```
Answer
--------------------------------------------------------------------------------
Employees must notify their manager and find shift coverage per the attendance
and sick-leave rules in the retrieved records. (Details follow the cited policy
lines.) A short Sources: line lists the record titles used.
```

Use `--dry-run` to print the composed prompt without calling the model, or `--show-context` to print the retrieved passages after the answer.

---

## Why This Matters

Store employees still rely on **manual lookup** for:

- **Recipes** — build steps, ratios, and modifiers scattered across bar and kitchen SOPs  
- **Policies** — benefits, attendance, conduct, and POS rules buried in long PDFs  
- **Procedures** — opening, cleaning, inspection, and incident flows that are easy to misremember under rush  

This system **compresses lookup from minutes to seconds**: one question, ranked evidence, and an optional **grounded** summary instead of tab-hunting or guessing under time pressure.

---

## Corpus

- **~100+ structured entries** in a typical single-store build; the **reference** `normalized_mosa_rag.jsonl` in this repo is **~130** lines after normalization.  
- **Includes:**  
  - **Drink recipes** — builds, tea volumes, modifiers, and bar execution detail  
  - **Employee policies** — handbook-style rules (attendance, conduct, benefits)  
  - **Operational procedures** — kitchen batch prep, cleaning, POS, events, and troubleshooting  

Each line is one **atomic record** with `retrieval_text` plus typed fields where applicable, designed for **dense retrieval** rather than raw page dumps.

---

## Architecture

End-to-end pipeline:

**`User query` → `BGE query embedding` → `FAISS similarity search` → `Top-K JSONL records` → `Prompt assembly` → `LLM reasoning` → `Answer + sources`**

| Stage | Implementation |
|--------|------------------|
| **Corpus** | Normalized **JSONL**: one record per line (recipe, procedure, policy chunk) with `retrieval_text` and structured fields for traceability. |
| **Embedding** | **`BAAI/bge-base-en-v1.5`** via **ONNX** (`onnxruntime`) and `transformers` tokenization; query uses the BGE retrieval instruction prefix by default. |
| **Similarity search** | **FAISS** `IndexFlatIP` over **L2-normalized** vectors so inner product matches **cosine similarity** ranking. |
| **LLM** | **Pluggable HTTP layer**: **`call_llm(prompt)`** targets **Ollama** `/api/generate` (local or remote base URL—same contract works behind a VM or container). Swap the adapter to wire a different provider without changing retrieval or prompt construction. |

Build path (PDFs → JSONL) lives under `mosa_rag/` (`build_corpus`, extractors, schema). **Level-1** `retrieve_pdf.py` offers the same BGE + FAISS stack over **raw PDF chunks** for experiments without the JSONL layer.

```
  JSONL corpus (N records)
           |
           v
  +------------------+
  | BGE ONNX encode  |  batch over retrieval_text
  +--------+---------+
           |
           v
  +------------------+
  | FAISS index      |  optional persist (.rag_index_cache)
  +--------+---------+
           ^   query vector
           |
  +--------+---------+
  | Top-K records    |  formatted context
  +--------+---------+
           |
           v
  +------------------+
  | LLM (optional)   |  Ollama HTTP; temperature 0 default
  +------------------+
```

---

## Results / Metrics

| Dimension | Value |
|-----------|--------|
| **Corpus scale** | **137** operational records in `normalized_mosa_rag.jsonl` (drink builds, prep rules, POS, policies, etc.). |
| **Retrieval quality (JSONL)** | **39 / 39** scored eval queries passed **Hit@1** and **Hit@5** with **MRR 1.000** across `eval_sets/mosa_rag_smoke.jsonl` (19 cases) and `mosa_rag_paraphrase.jsonl` (20 cases), `top_k=5`, BGE instruction on. |
| **Retrieval quality (PDF probe)** | **8 / 8** handbook questions: **Hit@1** and **Hit@3** at **MRR 1.000** with chunk size **80**, overlap **30** (`evaluate_retrieval.py`). |
| **End-to-end latency (cold index)** | **~52 s** full PDF eval run (embed index + 8 queries); **~61 s** JSONL eval run (embed **137** records + two eval files). First run includes model artifact load from Hugging Face. |
| **Operational impact** | Designed to shrink **multi-minute** PDF or chat “guesswork” into a **sub-minute** loop: one query, ranked evidence, optional grounded generation. |

Reproduce the JSONL numbers:

```bash
python evaluate_mosa_rag_jsonl.py --jsonl normalized_mosa_rag.jsonl --top-k 5
```

Reproduce the PDF numbers:

```bash
python evaluate_retrieval.py \
  --pdf-path "data/Mosa Employee Handbook.pdf" \
  --chunk-sizes 80 --chunk-overlaps 30 --top-k 3
```

---

## Design Decisions

- **JSONL over a database** — **Versionable**, **diff-friendly**, and **zero ops** for a resume-scale project: grep, `git`, and plain scripts can audit the corpus. Rows map 1:1 to retrieval units without schema migrations for early iteration.
- **BGE embeddings** — Strong **English dense retrieval** out of the box; ONNX path avoids a heavy PyTorch stack on resource-constrained machines while keeping quality competitive for short operational passages.
- **Retrieval before LLM** — The LLM never answers from a **blank context** in the main design: **constraints and citations** are derived from **forced top-k text**, reducing hallucinated policy or recipe steps.
- **Local vs cloud LLM** — **Ollama** keeps **PII and prompts on-network you control** (laptop, office LAN, or a **remote base URL** for the same HTTP API). The boundary is **environment-driven**, not a code fork—suitable for internships demonstrating **security-aware** design.

---

## Example Use Case

**Query:** *“What happens if I am sick?”*

1. **Embedding** — The question is encoded with the BGE retrieval instruction.  
2. **Search** — FAISS returns the **top-k** rows whose `retrieval_text` and fields best match sick leave and attendance policy.  
3. **Prompting** — `answer_mosa_rag_jsonl.py` builds a single prompt: system rules (**use only these records**), the question, and the **concatenated retrieved records** (titles, scores, text).  
4. **LLM** — `call_llm` sends that prompt to Ollama; the model produces a **short grounded answer** and a **Sources:** line tied to record titles.  
5. **Optional `--show-context`** — Prints the exact passages the model saw, for training and QA.

Dry-run (no LLM call):

```bash
export OLLAMA_PROVIDER=ollama_local
python answer_mosa_rag_jsonl.py "What happens if I am sick?" --dry-run
```

---

## Performance

| Phase | What you pay (typical laptop CPU) |
|--------|-----------------------------------|
| **Corpus embedding** | **O(N)** over all records—**~1 minute** class for **~140** rows in the captured eval (dominates cold start). |
| **Query embedding + search** | **Milliseconds to low seconds** per query after the index exists—single batch through ONNX + FAISS `search`. |
| **Persisted index** | **`load_or_build_faiss_index`** skips full re-embed when JSONL **mtime/size** and model id match cache metadata—**interactive retrieval** drops to **encode-query + disk index load + search**, i.e. **seconds**, not a full minute. |
| **LLM** | Bounded by local **Ollama** generation; short answers with **temperature 0** are usually **a few seconds to tens of seconds** depending on model and hardware. |

---

## Future Improvements

- **Vector tier** — Move from **flat FAISS** to **IVF / PQ** or a **managed vector database** (Pinecone, pgvector, Weaviate) when **N** grows past **thousands** of live records.  
- **Reranking** — Add a **cross-encoder** or lightweight reranker on the top **20 → 5** hits to sharpen boundary queries.  
- **Cloud deployment** — Package retrieval as a **small API** (e.g. **AWS Lambda** + object storage for index snapshots; **Cloud Run** + VPC for **remote Ollama**); keep the **HTTP** LLM boundary for swap-in observability.  
- **Benchmarks** — Expand **eval_sets** with harder **paraphrase** and **negative** probes; track **Hit@k** and **latency SLOs** in CI.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Search only (JSONL):**

```bash
python retrieve_mosa_rag_jsonl.py "How do I make a TGY special?" --jsonl normalized_mosa_rag.jsonl
```

**Grounded answer (Ollama running locally):**

```bash
export OLLAMA_PROVIDER=ollama_local
python answer_mosa_rag_jsonl.py "How do I make a TGY special?" --jsonl normalized_mosa_rag.jsonl
```

**PDF chunk baseline (no JSONL, no LLM):**

```bash
python retrieve_pdf.py path/to/document.pdf "What is the return policy?"
```

**BGE query instruction** (on by default for queries, off for stored passages):

`Represent this sentence for searching relevant passages: `  

Disable with `--raw-query` on the CLI tools.

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Runtime | Python 3 |
| PDF ingestion | `pypdf` |
| Tokenization | `transformers` |
| Embeddings | `BAAI/bge-base-en-v1.5` ONNX, `onnxruntime`, `huggingface-hub` |
| Vectors | `faiss-cpu`, `numpy` |
| LLM | Ollama `/api/generate` (`mosa_rag/llm.py`) |

---

## Corpus workflow

Build, validate, and detailed CLI flags: **`RAG_CORPUS_README.md`**.
