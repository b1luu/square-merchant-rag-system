#!/usr/bin/env python3
"""Minimal FastAPI server wrapping the existing RAG + Ollama pipeline."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mosa_rag.faiss_cache import default_cache_root, load_or_build_faiss_index
from mosa_rag.llm import call_llm
from mosa_rag.retrieve_jsonl import build_chunks, load_rows
from retrieve_pdf import MODEL_NAME, retrieve

from answer_mosa_rag_jsonl import build_context, build_user_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_PATH = Path(os.getenv("RAG_JSONL", "normalized_mosa_rag.jsonl"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Pre-load corpus + FAISS index once at startup
# ---------------------------------------------------------------------------
print(f"Loading corpus from {JSONL_PATH} ...", file=sys.stderr)
_rows = load_rows(JSONL_PATH)
_chunks = build_chunks(_rows, JSONL_PATH)
_cache_root = default_cache_root(JSONL_PATH)
_encoder, _index, _ = load_or_build_faiss_index(
    chunks=_chunks,
    model_name=MODEL_NAME,
    source_jsonl=JSONL_PATH,
    cache_root=_cache_root,
)
print(f"Ready — {len(_chunks)} records indexed.", file=sys.stderr)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_INSTRUCTIONS = (
    "You are answering internal Mosa Tea staff questions. "
    "Only use the provided retrieved records—treat them as the only source of truth. "
    "If the records are insufficient or conflicting, say so plainly. "
    "Never present general knowledge or guesses as if they came from the records."
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="message is required")

    logger.info("Query: %s", query)

    try:
        results = retrieve(
            encoder=_encoder,
            index=_index,
            chunks=_chunks,
            query=query,
            top_k=min(TOP_K, len(_chunks)),
            use_instruction=True,
        )
        context = build_context(_rows, results)
        user_input = build_user_input(query, context)
        prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{user_input}"

        logger.info("Calling Ollama ...")
        answer = call_llm(prompt)
        logger.info("Ollama responded (%d chars)", len(answer) if answer else 0)

        if not answer:
            raise HTTPException(status_code=502, detail="Empty response from model")

        return ChatResponse(response=answer)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat endpoint error: %s", traceback.format_exc())
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "records": str(len(_chunks))}
