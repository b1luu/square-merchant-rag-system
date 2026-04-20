#!/usr/bin/env python3
"""Minimal FastAPI server wrapping the existing RAG + Ollama pipeline."""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mosa_rag.llm import call_llm
from mosa_rag.runtime import ANSWER_TOP_K_DEFAULT, ResidentRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_PATH = Path(os.getenv("RAG_JSONL", "normalized_mosa_rag.jsonl"))
TOP_K = int(os.getenv("RAG_TOP_K", str(ANSWER_TOP_K_DEFAULT)))

# ---------------------------------------------------------------------------
# Pre-load corpus + FAISS index once at startup via ResidentRetriever
# ---------------------------------------------------------------------------
logger.info("Loading corpus from %s ...", JSONL_PATH)
_retriever = ResidentRetriever(jsonl=JSONL_PATH)
logger.info("Ready — %d records indexed.", len(_retriever.rows))

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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    confidence: dict | None = None
    verification: dict | None = None
    abstained: bool = False


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="message is required")

    logger.info("Query: %s", query)

    try:
        prompt, _prompt_ctx, _display_ctx, _results, confidence = _retriever.build_prompt_bundle(
            query, top_k=TOP_K
        )

        if confidence.should_abstain:
            logger.info("Abstaining — confidence level=%s score=%.4f", confidence.level, confidence.score)
            return ChatResponse(
                response="The retrieved records don't clearly answer this question.",
                confidence=asdict(confidence),
                verification=asdict(_retriever.verify_answer("", [])),
                abstained=True,
            )

        logger.info("Calling Ollama (confidence level=%s score=%.4f) ...", confidence.level, confidence.score)
        answer = call_llm(prompt)
        logger.info("Ollama responded (%d chars)", len(answer) if answer else 0)

        if not answer:
            raise HTTPException(status_code=502, detail="Empty response from model")

        return ChatResponse(
            response=answer,
            confidence=asdict(confidence),
            verification=asdict(_retriever.verify_answer(answer, _results)),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat endpoint error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "records": len(_retriever.rows),
        "top_k": TOP_K,
    }
