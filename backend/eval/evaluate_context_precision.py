from app.rag.retriever import search_documents
from eval.chunk_ground_truth import CHUNK_GROUND_TRUTH


def is_relevant(result, expected):
    payload = result["document"].payload

    if payload["filename"] != expected["expected_filename"]:
        return False

    result_start = payload["page_start"]
    result_end = payload["page_end"]

    expected_start, expected_end = expected["expected_pages"]

    return result_start <= expected_end and result_end >= expected_start


def precision_at_k(results, expected, k):
    top_results = results[:k]

    if not top_results:
        return 0

    relevant_count = sum(1 for result in top_results if is_relevant(result, expected))

    return relevant_count / len(top_results)


def evaluate_context_precision():

    precisions_at_1 = []
    precisions_at_3 = []
    precisions_at_5 = []

    print("\nCONTEXT PRECISION EVALUATION")
    print("=" * 80)

    for index, test in enumerate(CHUNK_GROUND_TRUTH, start=1):

        results = search_documents(
            query=test["query"],
            top_k=5,
        )

        p1 = precision_at_k(results, test, 1)

        p3 = precision_at_k(results, test, 3)

        p5 = precision_at_k(results, test, 5)

        precisions_at_1.append(p1)
        precisions_at_3.append(p3)
        precisions_at_5.append(p5)

        print(f"\n[{index}]")
        print(f"Query: {test['query']}")

        print(f"Precision@1: {p1:.2%}")
        print(f"Precision@3: {p3:.2%}")
        print(f"Precision@5: {p5:.2%}")

    avg_p1 = sum(precisions_at_1) / len(precisions_at_1)

    avg_p3 = sum(precisions_at_3) / len(precisions_at_3)

    avg_p5 = sum(precisions_at_5) / len(precisions_at_5)

    print("\n" + "=" * 80)

    print(f"Average Precision@1: {avg_p1:.2%}")
    print(f"Average Precision@3: {avg_p3:.2%}")
    print(f"Average Precision@5: {avg_p5:.2%}")


if __name__ == "__main__":
    evaluate_context_precision()
