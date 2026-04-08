# PDF Retrieval Only

This is a minimal level-1 retrieval pipeline:

1. Read a PDF with `pypdf`
2. Split it into overlapping chunks
3. Embed the chunks with `BAAI/bge-base-en-v1.5`
4. Store the embeddings in a FAISS index
5. Embed the query
6. Retrieve the top matching chunks
7. Print them out

No LLM is used here.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Default output is the top 3 chunks:

```bash
python retrieve_pdf.py path/to/document.pdf "What does this PDF say about pricing?"
```

If you want the top 5 instead:

```bash
python retrieve_pdf.py path/to/document.pdf "What does this PDF say about pricing?" --top-k 5
```

You can also tune chunking:

```bash
python retrieve_pdf.py path/to/document.pdf "Summarize the refund policy" --chunk-size 300 --chunk-overlap 75
```

## Why the query instruction is used

The BGE model card recommends adding this instruction to short retrieval queries:

`Represent this sentence for searching relevant passages: `

The script applies that instruction to the query by default and does not apply it to document chunks.

If you want to test retrieval without the instruction:

```bash
python retrieve_pdf.py path/to/document.pdf "What is the return policy?" --raw-query
```

## Notes

- `pypdf` extracts embedded text. If your PDF is scanned images, you need OCR first.
- FAISS is built in memory in this version. You can save/load the index later once you move beyond level 1.
- `BAAI/bge-base-en-v1.5` will download from Hugging Face the first time you run the script.
