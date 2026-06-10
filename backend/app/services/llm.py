import groq
import structlog
from fastapi import HTTPException

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert document analyst. Answer questions based ONLY on the provided document "
    "excerpts. If the answer is not in the context, say 'I couldn't find information about that "
    "in the document.' Never make up information."
)


async def generate_answer(
    question: str,
    chunks: list[str],
    model: str,
    api_key: str,
) -> dict:
    """Call the Groq API to generate an answer grounded in the provided chunks.

    Returns a dict with keys: answer, tokens_used, model.
    """
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    user_message = f"Document excerpts:\n{context}\n\nQuestion: {question}"

    client = groq.AsyncGroq(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
    except groq.RateLimitError as exc:
        logger.warning("llm.rate_limit", error=str(exc))
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded. Please retry shortly.")
    except groq.APIError as exc:
        logger.error("llm.api_error", error=str(exc))
        raise HTTPException(status_code=502, detail="LLM API error. Please try again later.")

    answer = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens if response.usage else 0
    logger.info("llm.generated", model=model, tokens_used=tokens_used)
    return {"answer": answer, "tokens_used": tokens_used, "model": model}
