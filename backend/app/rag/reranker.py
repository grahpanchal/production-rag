from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank_documents(query: str, documents: list, top_k: int = 5):
    if not documents:
        return []

    pairs = []

    for document in documents:
        text = document.payload.get("text", "")
        pairs.append([query, text])

    scores = model.predict(pairs)

    ranked_documents = []

    for document, score in zip(documents, scores):
        ranked_documents.append({"document": document, "rerank_score": float(score)})

    ranked_documents.sort(key=lambda x: x["rerank_score"], reverse=True)

    return ranked_documents[:top_k]
