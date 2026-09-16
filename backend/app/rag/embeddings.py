from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_embeddings(texts: list[str]) -> list[list[float]]:

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()