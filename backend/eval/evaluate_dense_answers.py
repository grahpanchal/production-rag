from app.config import settings
from app.db.qdrant import client
from app.rag.embeddings import create_embeddings
from app.rag.generator import generate_answer
from eval.answer_tests import ANSWER_TESTS


def dense_search(query: str, top_k: int = 5):
    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def evaluate_dense_answers():

    total = len(ANSWER_TESTS)
    passed = 0

    print("\nDENSE-ONLY ANSWER QUALITY EVALUATION")
    print("=" * 80)

    for index, test in enumerate(ANSWER_TESTS, start=1):

        documents = dense_search(query=test["query"], top_k=5)

        context_parts = []
        sources = []

        for source_index, document in enumerate(documents, start=1):

            payload = document.payload

            context_parts.append(f"""
[Source {source_index}]

Filename: {payload['filename']}
Pages: {payload['page_start']} - {payload['page_end']}

Content:
{payload['text']}
""")

            sources.append(
                {
                    "filename": payload["filename"],
                    "page_start": payload["page_start"],
                    "page_end": payload["page_end"],
                }
            )

        context = "\n\n".join(context_parts)

        result = generate_answer(question=test["query"], context=context)

        # Generator returns structured response:
        # {
        #     "answer": "...",
        #     "citations": [1, 3]
        # }
        if isinstance(result, dict):
            answer_text = result.get("answer", "")
            citations = result.get("citations", [])
        else:
            answer_text = result
            citations = []

        answer_lower = answer_text.lower()

        # Check whether at least one expected keyword is present.
        keyword_match = any(
            keyword.lower() in answer_lower for keyword in test["expected_keywords"]
        )

        # Check structured citations instead of looking for "[Source X]"
        has_citation = len(citations) > 0

        # Check whether retrieved context contains the expected document.
        source_match = any(
            source["filename"] == test["expected_filename"] for source in sources
        )

        passed_test = keyword_match and has_citation and source_match

        if passed_test:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{index}] {status}")
        print(f"Query: {test['query']}")
        print(f"Answer: {answer_text}")
        print(f"Expected keywords: {test['expected_keywords']}")
        print(f"Citations: {citations}")
        print(f"Citation present: {has_citation}")
        print(f"Correct source: {source_match}")

    score = passed / total

    print("\n" + "=" * 80)
    print(f"Dense-only Answer Quality: {score:.2%}")
    print(f"Passed: {passed}/{total}")


if __name__ == "__main__":
    evaluate_dense_answers()
