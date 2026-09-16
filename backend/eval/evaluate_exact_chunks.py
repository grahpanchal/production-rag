from app.rag.retriever import search_documents
from eval.chunk_id_ground_truth import CHUNK_ID_GROUND_TRUTH


def evaluate_exact_chunks():
    total = len(CHUNK_ID_GROUND_TRUTH)

    hits = 0
    reciprocal_ranks = []

    print("\nEXACT CHUNK-ID RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, test in enumerate(CHUNK_ID_GROUND_TRUTH, start=1):
        results = search_documents(
            query=test["query"],
            top_k=5,
        )

        retrieved_ids = [str(result["document"].id) for result in results]

        relevant_ids = set(test["relevant_ids"])

        first_relevant_rank = None

        for rank, point_id in enumerate(retrieved_ids, start=1):
            if point_id in relevant_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank is not None:
            hits += 1
            reciprocal_rank = 1 / first_relevant_rank
        else:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        status = "PASS" if first_relevant_rank else "FAIL"

        print(f"\n[{index}] {status}")
        print(f"Query: {test['query']}")

        print("Relevant IDs:", test["relevant_ids"])

        print("Retrieved IDs:", retrieved_ids)

        if first_relevant_rank:
            print(f"First relevant rank: " f"{first_relevant_rank}")

            print(f"Reciprocal Rank: " f"{reciprocal_rank:.2f}")
        else:
            print("First relevant rank: " "Not found in Top 5")

    recall_at_5 = hits / total
    mrr_at_5 = sum(reciprocal_ranks) / total

    print("\n" + "=" * 80)

    print(f"Exact Chunk Recall@5: " f"{recall_at_5:.2%}")

    print(f"Exact Chunk MRR@5:    " f"{mrr_at_5:.2%}")

    print(f"Passed:               " f"{hits}/{total}")


if __name__ == "__main__":
    evaluate_exact_chunks()
