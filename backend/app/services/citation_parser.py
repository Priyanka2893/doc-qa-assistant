import re
from dataclasses import dataclass

from app.services.confidence_scorer import ScoredChunk

_ABSTENTION_PREFIX = "Insufficient information"
_SOURCE_PATTERN = re.compile(r"\[Source (\d+)\]")


@dataclass
class ParsedCitation:
    tag: str
    source_number: int
    chunk: ScoredChunk | None
    text_excerpt: str
    page_number: int | None
    chunk_index: int | None
    confidence_score: float | None


@dataclass
class CitationResult:
    answer_with_citations: str
    citations: list[ParsedCitation]
    unmapped_citations: list[str]
    is_abstention: bool
    citation_coverage: float


def parse_citations(answer: str, chunks: list[ScoredChunk]) -> CitationResult:
    """Map [Source N] tags in the answer to actual retrieved chunks.

    Tags referencing non-existent indices go into unmapped_citations — a
    hallucination signal.
    """
    is_abstention = answer.strip().startswith(_ABSTENTION_PREFIX)

    seen: set[str] = set()
    citations: list[ParsedCitation] = []
    unmapped: list[str] = []

    for m in _SOURCE_PATTERN.finditer(answer):
        n = int(m.group(1))
        tag = f"[Source {n}]"
        if tag in seen:
            continue
        seen.add(tag)

        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            citations.append(
                ParsedCitation(
                    tag=tag,
                    source_number=n,
                    chunk=chunk,
                    text_excerpt=chunk.text[:200],
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    confidence_score=chunk.confidence.composite_score,
                )
            )
        else:
            unmapped.append(tag)

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    if sentences:
        cited_count = sum(1 for s in sentences if _SOURCE_PATTERN.search(s))
        coverage = cited_count / len(sentences)
    else:
        coverage = 0.0

    return CitationResult(
        answer_with_citations=answer,
        citations=citations,
        unmapped_citations=unmapped,
        is_abstention=is_abstention,
        citation_coverage=round(coverage, 4),
    )
