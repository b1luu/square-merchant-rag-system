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

### Retrieval quality (reproducible in this repo)

The evaluators report **Hit@1** (correct unit ranked first), **Hit@k** (correct unit in the top‑k list), and **MRR** (mean reciprocal rank of the first correct hit). Sweep chunk settings to compare configurations on the same PDF and question set.

**Example (bundled PDF eval):** `evaluate_retrieval.py` ships with **8** fixed questions against the PDF given by **`--pdf-path`** (see the script default). With `--chunk-sizes 80 --chunk-overlaps 30 --top-k 3` and the default BGE query instruction, one representative run produced:

| Eval setup | Indexed chunks | Hit@1 | Hit@k | MRR |
|------------|----------------|-------|-------|-----|
| 8 cases, instruction on | 60 | 8/8 | 8/8 | 1.000 |

Reproduce:

```bash
python evaluate_retrieval.py --chunk-sizes 80 --chunk-overlaps 30 --top-k 3
```

Different `--pdf-path`, cases, or `top-k` will change these numbers.

For the JSONL corpus path, use:

```bash
python evaluate_mosa_rag_jsonl.py --jsonl path/to/corpus.jsonl
```

That prints Hit@1, Hit@k, MRR per eval file and **by category** (see `eval_sets/`).

### Wall-clock latency (engineering, not “ops %”)

Compare end-to-end script time before and after a change (for example **cold** FAISS build vs **cached** index on the JSONL tools) with the shell `time` command. Report improvement as:

`(t_baseline − t_new) / t_baseline × 100%` when lower time is better.

### Operational impact (your numbers only)

Team-level outcomes—**% fewer wrong procedures**, **% shorter time-to-documented answer**, **ticket reopen rate**, audit scores—depend on **how** you roll out search/RAG and **what** you measure in HR, QA, or the helpdesk. This repository does **not** compute those automatically.

Use a simple table in internal reports (fill in after you have baselines):

| Metric | Baseline | After rollout | Δ % |
|--------|----------|---------------|-----|
| Median time to first grounded answer | | | |
| Weekly procedure / policy incidents | | | |
| Staff survey (clarity of procedures) | | | |

For any metric where **lower is better**, use  
`(baseline − new) / baseline × 100%`.  
For metrics where **higher is better**, use  
`(new − baseline) / baseline × 100%`.

**Do not invent** improvement percentages in this README; publish only values you have actually measured.

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
