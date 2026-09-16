from app.rag.retriever import search_documents
from app.rag.generator import generate_answer

NO_ANSWER = "I couldn't find the answer in the provided documents."


def answer_question(
    question: str,
    top_k: int = 5,
    filename: str | None = None,
    file_id: str | None = None,
):
    results = search_documents(
        query=question,
        top_k=top_k,
        filename=filename,
        file_id=file_id,
    )

    # Do not call the LLM if retrieval found nothing relevant
    if not results:
        return {
            "answer": NO_ANSWER,
            "sources": [],
        }

    context_parts = []

    for index, result in enumerate(results, start=1):
        payload = result["document"].payload

        context_parts.append(f"""
[Source {index}]

Filename: {payload['filename']}

Pages: {payload['page_start']} - {payload['page_end']}

Content:

{payload['text']}
""")

    context = "\n\n".join(context_parts)

    generated = generate_answer(
        question=question,
        context=context,
    )

    answer = generated["answer"]
    citation_numbers = generated.get("citations", [])

    sources = []

    for index, result in enumerate(results, start=1):
        payload = result["document"].payload

        sources.append(
            {
                "source": index,
                "filename": payload["filename"],
                "page_start": payload["page_start"],
                "page_end": payload["page_end"],
                "text": payload["text"],
                "qdrant_score": result["document"].score,
                "rerank_score": result["rerank_score"],
            }
        )

    # Validate citations returned by the LLM.
    valid_source_numbers = {source["source"] for source in sources}

    valid_citations = [
        citation for citation in citation_numbers if citation in valid_source_numbers
    ]

    # Add citations to the final answer.
    if valid_citations:
        citation_text = " ".join(f"[Source {citation}]" for citation in valid_citations)

        answer = f"{answer}\n\n{citation_text}"

    return {
        "answer": answer,
        "sources": sources,
    }
