import json
import logging

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Groq client
# --------------------------------------------------

client = Groq(
    api_key=settings.groq_api_key,
    timeout=30.0,
)


# --------------------------------------------------
# Structured response schema
# --------------------------------------------------

CITATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


# --------------------------------------------------
# Generate answer
# --------------------------------------------------


def generate_answer(question: str, context: str):

    prompt = f"""
You are a reliable document-based AI assistant.

Answer the user's question using ONLY the provided document context.

STRICT RULES:

1. Use only information present in the provided context.
2. Do not use outside knowledge.
3. Do not guess or invent information.
4. If the context does not contain enough information to answer the
   question, answer exactly:

I couldn't find the answer in the provided documents.

5. Keep the answer clear and concise.
6. Every factual claim must be supported by the provided context.
7. For every factual claim, include the source number that directly
   supports it in the citations array.
8. NEVER cite a source just because it is related to the topic.
9. NEVER cite an unrelated source.
10. Only include source numbers that directly support the answer.
11. Do not invent source numbers.
12. If multiple sources support the answer, include only the sources
    that are actually needed.
13. Preserve code commands exactly as they appear in the documents.
14. Do not modify, invent, or correct commands from the documents.
15. Return ONLY JSON matching the provided schema.
16. Do not put [Source X] citations inside the answer text.
17. Put source numbers ONLY inside the citations array.

DOCUMENT CONTEXT:

=========================
{context}
=========================

USER QUESTION:

{question}
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            reasoning_effort="low",
            reasoning_format="hidden",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_answer",
                    "schema": CITATION_RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
            max_completion_tokens=800,
        )

    except Exception:

        logger.exception("Groq generation failed")

        raise

    content = response.choices[0].message.content

    try:

        data = json.loads(content)

    except json.JSONDecodeError:

        logger.exception("Groq returned invalid JSON")

        raise ValueError("LLM returned invalid JSON")

    return data
