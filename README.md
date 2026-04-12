# Mosa Ops RAG System

**Purpose:** **Mosa Ops RAG System** is a **domain-specific RAG** stack for operations: staff answers should come from **retrieved** passages (SOPs, recipes, policies)—not from the model’s unconstrained general knowledge on the full RAG path. The **level-1** entrypoint here is **PDF chunk retrieval** (no LLM). The main ops path adds a **structured JSONL knowledge base**, **BGE + FAISS** retrieval, an optional **on-disk index cache**, and an optional **local LLM** (Ollama) grounded in those hits.

---

## Tech stack

| Layer | Technology |
|--------|------------|
| Language | Python 3 |
| PDF text | `pypdf` |
| Tokenization | `transformers` (`AutoTokenizer`) |
| Embedding model | `BAAI/bge-base-en-v1.5` (ONNX export from Hugging Face) |
| Embedding inference | `onnxruntime` (CPU; model files pulled via `huggingface-hub`) |
| Vector search | `faiss-cpu` (`IndexFlatIP`, L2-normalized vectors for cosine-style ranking) |
| Numerics | `numpy` |
| LLM (full RAG only, not used by `retrieve_pdf.py`) | Ollama or any host exposing the same `/api/generate` HTTP API |

---

## Architecture (text diagram)

**Level 1 — PDF retrieval (implemented in `retrieve_pdf.py`):**

```
  +----------------------+
  | PDF file(s) / folder |
  +----------+-----------+
             |
             v
  +----------------------+
  | pypdf: text per page |
  +----------+-----------+
             |
             v
  +----------------------+
  | Word windows + overlap|
  | -> Chunk[]           |
  +----------+-----------+
             |
             v
  +----------------------+     +------------------------+
  | Hugging Face snapshot|     | Tokenize batches       |
  | ONNX model.onnx      |---->| onnxruntime inference  |
  +----------------------+     +------------+-----------+
                                            |
                                            v
                               +------------------------+
                               | L2-normalized vectors  |
                               +------------+-----------+
                                            |
                                            v
                               +------------------------+
                               | FAISS IndexFlatIP      |
                               | (built in memory)      |
                               +------------+-----------+
                                            ^
                                            |
                               +------------+-----------+
                               | Query string           |
                               | + optional BGE        |
                               |   query instruction    |
                               +------------+-----------+
                                            |
                                            v
                               +------------------------+
                               | Top-k chunk scores     |
                               | + source metadata      |
                               +------------------------+
```

**Typical extension — full RAG (see `RAG_CORPUS_README.md` for commands):**

```
  +----------------------+       +----------------------+
  | Curated corpus build |  -->  | Searchable store     |
  | (PDFs -> records)    |       | (e.g. JSONL rows)    |
  +----------------------+       +----------+-----------+
                                              |
                                              v
                               +------------------------+       +------------------+
                               | Same BGE + FAISS idea|  -->  | Optional: save   |
                               | over record texts    |       | FAISS to disk    |
                               +----------+-----------+       +------------------+
                                              |
                                              v
                               +------------------------+       +------------------+
                               | Compose prompt with  |  -->  | POST /api/       |
                               | top-k retrieved text   |       | generate (e.g.   |
                               +------------------------+       | Ollama)          |
                                                                 +------------------+
```

---

## PDF retrieval (level 1)

Minimal **retrieval-only** pipeline:

1. Read a PDF with `pypdf`  
2. Split into overlapping word chunks  
3. Embed chunks with `BAAI/bge-base-en-v1.5` (ONNX)  
4. Store vectors in a FAISS index  
5. Embed the query and retrieve top matches  
6. Print chunks (no LLM in this script)  

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default: top 3 chunks (defaults `--chunk-size 80 --chunk-overlap 30`):

```bash
python retrieve_pdf.py path/to/document.pdf "What does this document say about pricing?"
```

Index a directory of PDFs:

```bash
python retrieve_pdf.py path/to/pdf_folder "What is the policy on outside food in the kitchen?"
```

Top 5 matches:

```bash
python retrieve_pdf.py path/to/document.pdf "What does this document say about pricing?" --top-k 5
```

Tune chunking:

```bash
python retrieve_pdf.py path/to/document.pdf "Summarize the refund policy" --chunk-size 300 --chunk-overlap 75
```

Optional retrieval sweep (`evaluate_retrieval.py`):

```bash
python evaluate_retrieval.py --pdf-path path/to/document.pdf
```

```bash
python evaluate_retrieval.py --pdf-path path/to/document.pdf --chunk-sizes 60,80,100 --chunk-overlaps 10,20
```

## Quantifying results

Metrics below were produced by **running the evaluators in this repository** (same commands, fresh FAISS build each run). **Hit@1** = first hit is correct; **Hit@k** = correct row appears in top‑k; **MRR** = mean reciprocal rank of the first correct hit. Re-run after you change PDFs, JSONL, eval cases, or embedding settings—numbers will move if the data moves.

### PDF handbook retrieval (`evaluate_retrieval.py`)

Command:

```bash
python evaluate_retrieval.py \
  --pdf-path "data/Mosa Employee Handbook.pdf" \
  --chunk-sizes 80 \
  --chunk-overlaps 30 \
  --top-k 3
```

**Captured output (2026-04-12, repo checkout):**

| Cases | Chunk size / overlap | Chunks built | Hit@1 | Hit@3 | MRR | Wall time (approx.) |
|------:|---------------------|-------------:|------:|------:|----:|---------------------:|
| 8 | 80 / 30 | 60 | 8/8 | 8/8 | 1.000 | ~52 s |

Script header from that run: `PDF: data/Mosa Employee Handbook.pdf`, `Cases: 8`, `Top-k: 3`, `Instruction: True`, table row `80  30  60  8/8  8/8  1.000`.

Sweep more `(chunk_size, chunk_overlap)` pairs with `--chunk-sizes` / `--chunk-overlaps` to compare grids on the same PDF.

### JSONL corpus retrieval (`evaluate_mosa_rag_jsonl.py`)

Command (defaults: `eval_sets/mosa_rag_smoke.jsonl` + `eval_sets/mosa_rag_paraphrase.jsonl`, BGE query instruction on):

```bash
python evaluate_mosa_rag_jsonl.py --jsonl normalized_mosa_rag.jsonl --top-k 5
```

**Captured output (2026-04-12, same checkout):**

| Corpus | Records | Eval file | Scored cases | Hit@1 | Hit@5 | MRR |
|--------|--------:|-----------|-------------:|------:|------:|----:|
| `normalized_mosa_rag.jsonl` | 137 | `mosa_rag_smoke.jsonl` | 19 | 19/19 | 19/19 | 1.000 |
| same | 137 | `mosa_rag_paraphrase.jsonl` | 20 | 20/20 | 20/20 | 1.000 |

*Wall time for that full command (both files, cold FAISS build over 137 records): ~61 s on the machine used for the capture.*

That run reported **no misses** and **no “not rank 1”** rows for either file. Every **By category** line in the log showed **Hit@1 = Hit@5 = case count** and **MRR 1.000**.

Other files under `eval_sets/` (for example gap probes) are **not** in the default invocation; pass them explicitly if you want those numbers in the log.

### Wall-clock / engineering %

Compare script runtime before and after a change (e.g. cold embed vs cached index on the JSONL tools) with `time python …`. Improvement when lower is better:

`(t_baseline − t_new) / t_baseline × 100%`.

### Operational KPIs (fill in yourself)

Store-level **% improvements** (time to answer, incidents, survey scores) are **not** emitted by these scripts. Measure in your own tools, then use  
`(baseline − new) / baseline × 100%` when lower is better, or  
`(new − baseline) / baseline × 100%` when higher is better. **Do not** copy made-up business percentages into this file.

## Query instruction (BGE)

Short retrieval queries use the BGE-style prefix:

`Represent this sentence for searching relevant passages: `

The script applies that instruction to the query by default and does not apply it to document chunks.

To test retrieval without the instruction:

```bash
python retrieve_pdf.py path/to/document.pdf "What is the return policy?" --raw-query
```

## Notes

- `pypdf` uses embedded text; scanned PDFs need OCR first.  
- In `retrieve_pdf.py`, FAISS is built **in memory** each run; the JSONL scripts can persist an index under `.rag_index_cache/` (see **`RAG_CORPUS_README.md`**).  
- `BAAI/bge-base-en-v1.5` and `onnx/model.onnx` download from Hugging Face on first use.  
- This project uses `onnxruntime` for BGE inference instead of a PyTorch `sentence-transformers` runtime.  
- Results include source PDF file names when you index multiple documents together.  

## Mosa Ops RAG System — corpus workflow

For build, validate, search, and optional Ollama answers over the ops knowledge base, see **`RAG_CORPUS_README.md`**.
