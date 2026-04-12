# PDF retrieval (level 1)

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

Default: top 3 chunks:

```bash
python retrieve_pdf.py path/to/document.pdf "What does this document say about pricing?"
```

You can index a directory of PDFs:

```bash
python retrieve_pdf.py path/to/pdf_folder "What is the policy on outside food in the kitchen?"
```

More matches:

```bash
python retrieve_pdf.py path/to/document.pdf "Summarize the refund policy" --top-k 5
```

Tune chunking:

```bash
python retrieve_pdf.py path/to/document.pdf "Summarize the refund policy" --chunk-size 300 --chunk-overlap 75
```

Optional evaluation driver (point at your own PDF):

```bash
python evaluate_retrieval.py --pdf-path path/to/document.pdf
```

```bash
python evaluate_retrieval.py --pdf-path path/to/document.pdf --chunk-sizes 60,80,100 --chunk-overlaps 10,20
```

## Query instruction (BGE)

Short retrieval queries use the BGE-style prefix:

`Represent this sentence for searching relevant passages: `

Applied to the query by default, not to document chunks. To disable:

```bash
python retrieve_pdf.py path/to/document.pdf "What is the return policy?" --raw-query
```

## Notes

- `pypdf` uses embedded text; scanned PDFs need OCR first.  
- FAISS is built in memory in `retrieve_pdf.py`; the JSONL answer/search scripts can persist an index under `.rag_index_cache/` (see `RAG_CORPUS_README.md`).  
- The embedding model downloads from Hugging Face on first use.  
- This project uses `onnxruntime` for BGE inference.  
- Results include source file names when you index multiple PDFs together.  

## JSONL RAG (retrieval + optional local LLM)

For a normalized JSONL workflow (corpus build, validate, search, optional Ollama answers), see **`RAG_CORPUS_README.md`**.
