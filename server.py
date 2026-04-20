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
from mosa_rag.validation_agent import validate_answer
from serve_mosa_rag import MAX_QUERY_CHARS, MAX_TOP_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_PATH = Path(os.getenv("RAG_JSONL", "normalized_mosa_rag.jsonl"))


def _resolve_top_k_env() -> int:
    raw = os.getenv("RAG_TOP_K", str(ANSWER_TOP_K_DEFAULT))
    try:
        top_k = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"RAG_TOP_K must be an integer between 1 and {MAX_TOP_K}") from exc
    if top_k < 1 or top_k > MAX_TOP_K:
        raise RuntimeError(f"RAG_TOP_K must be an integer between 1 and {MAX_TOP_K}")
    return top_k


TOP_K = _resolve_top_k_env()

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
    answer_mode: str = "llm"
    confidence: dict | None = None
    verification: dict | None = None
    validation: dict | None = None
    abstained: bool = False


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="message is required")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"message must be {MAX_QUERY_CHARS} characters or fewer")

    logger.info("Query: %s", query)

    try:
        prompt, _prompt_ctx, _display_ctx, _results, confidence = _retriever.build_prompt_bundle(
            query, top_k=TOP_K
        )

        if confidence.should_abstain:
            logger.info("Abstaining — confidence level=%s score=%.4f", confidence.level, confidence.score)
            verification = _retriever.verify_answer("", [])
            return ChatResponse(
                response="The retrieved records don't clearly answer this question.",
                answer_mode="abstain",
                confidence=asdict(confidence),
                verification=asdict(verification),
                validation=asdict(
                    validate_answer(
                        confidence=confidence,
                        verification=verification,
                        answer_mode="abstain",
                        abstained=True,
                    )
                ),
                abstained=True,
            )

        if not os.getenv("OLLAMA_PROVIDER", "").strip():
            answer = _retriever.build_extractive_answer(_results)
            verification = _retriever.verify_answer(answer, _results)
            return ChatResponse(
                response=answer,
                answer_mode="extractive",
                confidence=asdict(confidence),
                verification=asdict(verification),
                validation=asdict(
                    validate_answer(
                        confidence=confidence,
                        verification=verification,
                        answer_mode="extractive",
                        abstained=False,
                    )
                ),
            )

        logger.info("Calling Ollama (confidence level=%s score=%.4f) ...", confidence.level, confidence.score)
        answer = call_llm(prompt)
        logger.info("Ollama responded (%d chars)", len(answer) if answer else 0)

        if not answer:
            raise HTTPException(status_code=502, detail="Empty response from model")

        verification = _retriever.verify_answer(answer, _results)
        return ChatResponse(
            response=answer,
            answer_mode="llm",
            confidence=asdict(confidence),
            verification=asdict(verification),
            validation=asdict(
                validate_answer(
                    confidence=confidence,
                    verification=verification,
                    answer_mode="llm",
                    abstained=False,
                )
            ),
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
