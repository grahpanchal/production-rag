from app.rag.retriever import search_documents
from eval.test_queries import TEST_QUERIES


def inspect_chunks():
    print("\nCHUNK-LEVEL RETRIEVAL INSPECTION")
    print("=" * 80)

    for index, test in enumerate(TEST_QUERIES, start=1):

        results = search_documents(query=test["query"], top_k=5)

        print(f"\n{'=' * 80}")
        print(f"QUERY {index}: {test['query']}")
        print(f"{'=' * 80}")

        for rank, result in enumerate(results, start=1):

            document = result["document"]
            payload = document.payload

            text = payload["text"].replace("\n", " ")

            if len(text) > 250:
                text = text[:250] + "..."

            print(f"\nRank: {rank}")
            print(f"Point ID: {document.id}")
            print(f"Filename: {payload['filename']}")
            print(f"Pages: " f"{payload['page_start']}-{payload['page_end']}")
            print(f"Qdrant score: " f"{document.score:.4f}")
            print(f"Rerank score: " f"{result['rerank_score']:.4f}")
            print(f"Text: {text}")


if __name__ == "__main__":
    inspect_chunks()
