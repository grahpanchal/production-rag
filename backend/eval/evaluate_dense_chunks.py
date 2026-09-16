from app.db.qdrant import client
from app.config import settings
from app.rag.embeddings import create_embeddings
from eval.chunk_ground_truth import CHUNK_GROUND_TRUTH


def search_dense(query: str, top_k: int = 5):
    query_embedding = create_embeddings([query])[0]

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def is_relevant(result, expected):
    payload = result.payload

    if payload["filename"] != expected["expected_filename"]:
        return False

    result_start = payload["page_start"]
    result_end = payload["page_end"]

    expected_start, expected_end = expected["expected_pages"]

    return result_start <= expected_end and result_end >= expected_start


def evaluate_dense_chunks():
    total = len(CHUNK_GROUND_TRUTH)

    hits = 0
    reciprocal_ranks = []

    print("\nDENSE-ONLY CHUNK RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, test in enumerate(CHUNK_GROUND_TRUTH, start=1):
        results = search_dense(query=test["query"], top_k=5)

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if is_relevant(result, test):
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

        print(
            f"Expected: "
            f"{test['expected_filename']} "
            f"pages {test['expected_pages'][0]}-"
            f"{test['expected_pages'][1]}"
        )

        for rank, result in enumerate(results, start=1):
            payload = result.payload

            print(
                f"Rank {rank}: "
                f"{payload['filename']} "
                f"pages "
                f"{payload['page_start']}-"
                f"{payload['page_end']} "
                f"dense={result.score:.4f}"
            )

        if first_relevant_rank:
            print(f"First relevant chunk: " f"Rank {first_relevant_rank}")
            print(f"Reciprocal Rank: " f"{reciprocal_rank:.2f}")
        else:
            print("First relevant chunk: " "Not found in Top 5")

    recall_at_5 = hits / total
    mrr_at_5 = sum(reciprocal_ranks) / total

    print("\n" + "=" * 80)

    print(f"Chunk Recall@5: " f"{recall_at_5:.2%}")

    print(f"Chunk MRR@5:    " f"{mrr_at_5:.2%}")

    print(f"Passed:         " f"{hits}/{total}")


if __name__ == "__main__":
    evaluate_dense_chunks()
