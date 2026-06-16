from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Correction:
    correction_id: str
    doc_id: str
    original_question: str
    original_answer: str
    corrected_answer: str
    question_embedding: list[float]
    submitted_at: datetime
    use_count: int = 0


class CorrectionStore:
    def __init__(self) -> None:
        self._corrections: list[Correction] = []

    async def add_correction(
        self,
        doc_id: str,
        question: str,
        original_answer: str,
        corrected_answer: str,
        embedder,
    ) -> str:
        loop = asyncio.get_event_loop()
        embedding: list[float] = await loop.run_in_executor(None, embedder.encode_query, question)
        correction = Correction(
            correction_id=str(uuid.uuid4()),
            doc_id=doc_id,
            original_question=question,
            original_answer=original_answer,
            corrected_answer=corrected_answer,
            question_embedding=embedding,
            submitted_at=datetime.utcnow(),
        )
        self._corrections.append(correction)
        logger.info("correction.added", correction_id=correction.correction_id, doc_id=doc_id)
        return correction.correction_id

    async def find_correction(
        self,
        question: str,
        doc_id: str,
        embedder,
        threshold: float = 0.92,
    ) -> Correction | None:
        doc_corrections = [c for c in self._corrections if c.doc_id == doc_id]
        if not doc_corrections:
            return None
        loop = asyncio.get_event_loop()
        q_emb: list[float] = await loop.run_in_executor(None, embedder.encode_query, question)
        q = np.array(q_emb)
        best_score = 0.0
        best: Correction | None = None
        for c in doc_corrections:
            c_vec = np.array(c.question_embedding)
            score = float(np.dot(q, c_vec) / (np.linalg.norm(q) * np.linalg.norm(c_vec) + 1e-9))
            if score > best_score:
                best_score, best = score, c
        if best is not None and best_score >= threshold:
            best.use_count += 1
            logger.info(
                "correction.matched",
                correction_id=best.correction_id,
                score=round(best_score, 4),
            )
            return best
        return None

    def list_corrections(self, doc_id: str | None = None) -> list[Correction]:
        if doc_id:
            return [c for c in self._corrections if c.doc_id == doc_id]
        return list(self._corrections)


_correction_store = CorrectionStore()


def get_correction_store() -> CorrectionStore:
    return _correction_store
