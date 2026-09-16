from app.rag.retriever import search_documents
from eval.chunk_ground_truth import CHUNK_GROUND_TRUTH


def is_relevant(result, expected):
    payload = result["document"].payload

    if payload["filename"] != expected["expected_filename"]:
        return False

    result_start = payload["page_start"]
    result_end = payload["page_end"]

    expected_start, expected_end = expected["expected_pages"]

    # Check whether page ranges overlap
    return result_start <= expected_end and result_end >= expected_start


def evaluate():
    total = len(CHUNK_GROUND_TRUTH)

    hits = 0
    reciprocal_ranks = []

    print("\nCHUNK-LEVEL RETRIEVAL EVALUATION")
    print("=" * 70)

    for index, test in enumerate(CHUNK_GROUND_TRUTH, start=1):

        results = search_documents(
            query=test["query"],
            top_k=5,
        )

        rank = None

        for position, result in enumerate(results, start=1):

            if is_relevant(result, test):
                rank = position
                break

        if rank is not None:
            hits += 1
            reciprocal_rank = 1 / rank
        else:
            reciprocal_rank = 0

        reciprocal_ranks.append(reciprocal_rank)

        status = "PASS" if rank is not None else "FAIL"

        print(f"\n[{index}] {status}")
        print(f"Query: {test['query']}")

        print(
            f"Expected: "
            f"{test['expected_filename']} "
            f"pages {test['expected_pages'][0]}-"
            f"{test['expected_pages'][1]}"
        )

        for position, result in enumerate(results, start=1):

            payload = result["document"].payload

            print(
                f"Rank {position}: "
                f"{payload['filename']} "
                f"pages {payload['page_start']}-"
                f"{payload['page_end']} "
                f"rerank={result['rerank_score']:.3f}"
            )

        if rank is not None:
            print(f"First relevant chunk: Rank {rank}")
            print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        else:
            print("First relevant chunk: Not found")

    recall_at_5 = hits / total
    mrr_at_5 = sum(reciprocal_ranks) / total

    print("\n" + "=" * 70)
    print(f"Chunk Recall@5: {recall_at_5:.2%}")
    print(f"Chunk MRR@5:    {mrr_at_5:.2%}")
    print(f"Passed:         {hits}/{total}")


if __name__ == "__main__":
    evaluate()

# from app.rag.retriever import search_documents
# from eval.chunk_ground_truth import CHUNK_GROUND_TRUTH


# def evaluate_chunks():

#     total = len(CHUNK_GROUND_TRUTH)

#     hits = 0
#     reciprocal_ranks = []

#     print("\nCHUNK-LEVEL RETRIEVAL EVALUATION")
#     print("=" * 80)

#     for index, test in enumerate(CHUNK_GROUND_TRUTH, start=1):

#         results = search_documents(query=test["query"], top_k=5)

#         retrieved_ids = [str(result["document"].id) for result in results]

#         relevant_ids = set(test["relevant_ids"])

#         first_relevant_rank = None

#         for rank, point_id in enumerate(retrieved_ids, start=1):
#             if point_id in relevant_ids:
#                 first_relevant_rank = rank
#                 break

#         if first_relevant_rank is not None:
#             hits += 1
#             reciprocal_rank = 1 / first_relevant_rank
#         else:
#             reciprocal_rank = 0

#         reciprocal_ranks.append(reciprocal_rank)

#         status = "PASS" if first_relevant_rank else "FAIL"

#         print(f"\n[{index}] {status}")
#         print(f"Query: {test['query']}")

#         print("Relevant IDs:", test["relevant_ids"])

#         print("Retrieved IDs:", retrieved_ids)

#         if first_relevant_rank:
#             print("First relevant rank:", first_relevant_rank)
#             print(f"Reciprocal Rank: " f"{reciprocal_rank:.2f}")
#         else:
#             print("First relevant rank: " "Not found in Top 5")

#     recall_at_5 = hits / total

#     mrr_at_5 = sum(reciprocal_ranks) / total

#     print("\n" + "=" * 80)

#     print(f"Chunk Recall@5: " f"{recall_at_5:.2%}")

#     print(f"Chunk MRR@5:    " f"{mrr_at_5:.2%}")

#     print(f"Passed:         " f"{hits}/{total}")


# if __name__ == "__main__":
#     evaluate_chunks()
