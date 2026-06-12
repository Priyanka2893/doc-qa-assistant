import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import database
from app.models import AskRequest, EvalSummary, ResponseMode, SearchMode
from app.services.citation_parser import parse_citations
from app.services.confidence_scorer import summarize_evidence_quality
from app.services.evaluator import evaluate_response
from app.services.hallucination_guard import pre_generation_gate, verify_answer
from app.services.llm import generate_answer
from app.services.prompt_builder import build_messages
from app.services.retriever import retrieve

logger = structlog.get_logger(__name__)
router = APIRouter()


class BenchmarkCase(BaseModel):
    question: str
    expected_keywords: list[str]
    should_abstain: bool = False


class BenchmarkRequest(BaseModel):
    doc_id: str
    test_cases: list[BenchmarkCase]


class BenchmarkCaseResult(BaseModel):
    question: str
    keywords_found: int
    total_keywords: int
    recall: float
    overall_score: float
    is_abstention: bool
    passed: bool


class BenchmarkResponse(BaseModel):
    total_cases: int
    passed: int
    avg_recall: float
    avg_overall_score: float
    results: list[BenchmarkCaseResult]


@router.get("/eval/summary", response_model=EvalSummary)
async def get_eval_summary(hours: int = 24) -> EvalSummary:
    """Aggregated evaluation metrics for the last N hours."""
    data = await database.get_eval_summary(hours)
    return EvalSummary(**data)


@router.get("/eval/document/{doc_id}", response_model=EvalSummary)
async def get_doc_eval_summary(doc_id: str) -> EvalSummary:
    """Per-document evaluation summary across all queries against that document."""
    doc = await database.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    data = await database.get_doc_eval_summary(doc_id)
    return EvalSummary(**data)


@router.post("/eval/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: Request, body: BenchmarkRequest) -> BenchmarkResponse:
    """Run a benchmark test suite against an uploaded document."""
    settings = request.app.state.settings
    qdrant_client = request.app.state.qdrant_client

    doc = await database.get_document(body.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{body.doc_id}' not found.")

    results: list[BenchmarkCaseResult] = []

    for case in body.test_cases:
        output = await retrieve(
            question=case.question,
            doc_id=body.doc_id,
            top_k=5,
            mode=SearchMode.HYBRID,
            rerank=True,
            qdrant_client=qdrant_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding_model=settings.EMBEDDING_MODEL,
            min_confidence=settings.MIN_CONFIDENCE_THRESHOLD,
            confidence_weights=settings.CONFIDENCE_WEIGHTS,
        )
        chunks = output.chunks

        gate_result = pre_generation_gate(chunks, settings.PRE_GEN_CONFIDENCE_GATE, settings.MIN_RAW_VECTOR_SCORE)
        if not gate_result.passed:
            from app.services.hallucination_guard import VerificationResult
            dummy_verification = VerificationResult(
                sentences=[],
                hallucination_risk=0.0,
                ungrounded_sentences=[],
                is_high_risk=False,
                grounded_count=0,
                ungrounded_count=0,
            )
            eval_metrics = await evaluate_response(
                question=case.question,
                chunks=chunks,
                answer="Insufficient information",
                verification_result=dummy_verification,
                is_abstention=True,
            )
            answer_text = "Insufficient information"
            is_abstention = True
        else:
            messages = build_messages(case.question, chunks, mode=ResponseMode.PLAIN)
            llm_result = await generate_answer(
                messages=messages,
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.1,
            )
            answer_text = llm_result["answer"]

            citation_result = parse_citations(answer_text, chunks)
            is_abstention = citation_result.is_abstention

            verification = await verify_answer(
                answer_text,
                chunks,
                token_fast_path_threshold=settings.POST_GEN_TOKEN_FAST_PATH,
                semantic_threshold=settings.POST_GEN_OVERLAP_THRESHOLD,
                high_risk_threshold=settings.HIGH_RISK_THRESHOLD,
            )
            eval_metrics = await evaluate_response(
                question=case.question,
                chunks=chunks,
                answer=answer_text,
                verification_result=verification,
                is_abstention=is_abstention,
            )

        answer_lower = answer_text.lower()
        found = sum(1 for kw in case.expected_keywords if kw.lower() in answer_lower)
        total = len(case.expected_keywords)
        recall = round(found / total, 4) if total > 0 else 1.0
        passed = recall >= 0.5 or (case.should_abstain and is_abstention)

        results.append(BenchmarkCaseResult(
            question=case.question,
            keywords_found=found,
            total_keywords=total,
            recall=recall,
            overall_score=eval_metrics.overall_score,
            is_abstention=is_abstention,
            passed=passed,
        ))

    total_cases = len(results)
    passed_count = sum(1 for r in results if r.passed)
    avg_recall = round(sum(r.recall for r in results) / total_cases, 4) if total_cases else 0.0
    avg_overall = round(sum(r.overall_score for r in results) / total_cases, 4) if total_cases else 0.0

    logger.info(
        "eval.benchmark",
        doc_id=body.doc_id,
        total_cases=total_cases,
        passed=passed_count,
        avg_recall=avg_recall,
        avg_overall_score=avg_overall,
    )

    return BenchmarkResponse(
        total_cases=total_cases,
        passed=passed_count,
        avg_recall=avg_recall,
        avg_overall_score=avg_overall,
        results=results,
    )
