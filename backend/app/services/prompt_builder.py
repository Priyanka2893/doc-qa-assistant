from app.models import ResponseMode
from app.services.confidence_scorer import ScoredChunk

_CITED_SYSTEM = """\
You are a precise document analyst. Your ONLY source of knowledge is the context provided below.

STRICT RULES:
1. Every factual claim MUST be followed by a citation tag: [Source 1], [Source 2], etc.
2. Use ONLY information from the provided context. Do NOT use any outside knowledge.
3. If a fact appears in multiple sources, cite the most relevant one.
4. If the context does not contain the answer, respond with exactly: \
"Insufficient information in the provided documents."
5. Do NOT make assumptions, inferences beyond what is stated, or fill gaps with general knowledge.
6. Format: answer in clear prose with inline [Source N] tags after each claim.

Context:
{context}"""

_PLAIN_SYSTEM = """\
You are a precise document analyst. Your ONLY source of knowledge is the context provided below.

STRICT RULES:
1. Answer using ONLY information from the provided context. Do NOT use any outside knowledge.
2. If the context does not contain the answer, respond with exactly: \
"Insufficient information in the provided documents."
3. Be concise and direct.

Context:
{context}"""

_STRICT_ABSTAIN_SYSTEM = """\
You are a cautious document analyst operating in strict verification mode.

STRICT RULES:
1. You MUST only answer if the context contains explicit, clear information answering the question.
2. If there is ANY ambiguity or the answer is not explicitly stated, respond with exactly: \
"Insufficient information in the provided documents."
3. Every factual claim MUST be followed by a citation tag: [Source 1], [Source 2], etc.
4. Do NOT use any outside knowledge.

Context:
{context}"""

_PROMPTS: dict[ResponseMode, str] = {
    ResponseMode.CITED: _CITED_SYSTEM,
    ResponseMode.PLAIN: _PLAIN_SYSTEM,
    ResponseMode.STRICT_ABSTAIN: _STRICT_ABSTAIN_SYSTEM,
}


def get_system_prompt(mode: ResponseMode) -> str:
    return _PROMPTS[mode]


def format_context(scored_chunks: list[ScoredChunk]) -> str:
    parts = []
    for i, chunk in enumerate(scored_chunks, 1):
        header = (
            f"[Source {i}] "
            f"(File: {chunk.filename}, "
            f"Page: {chunk.page_number or 'N/A'}, "
            f"Confidence: {chunk.confidence.composite_score:.2f})"
        )
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    question: str,
    scored_chunks: list[ScoredChunk],
    mode: ResponseMode = ResponseMode.CITED,
) -> list[dict]:
    """Build the messages array for the LLM API call."""
    context = format_context(scored_chunks)
    system = get_system_prompt(mode).replace("{context}", context)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]
