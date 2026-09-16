from app.rag.retriever import search_documents
from eval.test_queries import TEST_QUERIES


def evaluate():
    total = len(TEST_QUERIES)
    hits = 0
    reciprocal_ranks = []

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for index, test in enumerate(TEST_QUERIES, start=1):

        results = search_documents(query=test["query"], top_k=5)

        filenames = [result["document"].payload["filename"] for result in results]

        expected = test["expected_filename"]

        # Find first relevant result
        rank = None

        for position, filename in enumerate(filenames, start=1):
            if filename == expected:
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
        print(f"Expected: {expected}")
        print(f"Retrieved: {filenames}")

        if rank:
            print(f"First relevant result: Rank {rank}")
            print(f"Reciprocal Rank: {reciprocal_rank:.2f}")
        else:
            print("First relevant result: Not found")

    recall_at_5 = hits / total
    mrr_at_5 = sum(reciprocal_ranks) / total

    print("\n" + "=" * 70)
    print(f"Recall@5: {recall_at_5:.2%}")
    print(f"MRR@5:    {mrr_at_5:.2%}")
    print(f"Passed:   {hits}/{total}")


if __name__ == "__main__":
    evaluate()
