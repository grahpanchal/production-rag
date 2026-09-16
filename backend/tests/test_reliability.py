from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(
    app,
    raise_server_exceptions=False,
)

AUTH_HEADERS = {
    "X-API-Key": settings.api_key,
}


def test_upload_rejects_file_over_10mb():
    large_pdf = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.pdf",
                large_pdf,
                "application/pdf",
            )
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 413

    data = response.json()

    assert data["detail"] == "File size must be 10 MB or less"


def test_search_handles_qdrant_failure(monkeypatch):

    def failing_search_documents(
        query,
        top_k=5,
        filename=None,
        file_id=None,
    ):
        raise RuntimeError("Qdrant connection failed")

    monkeypatch.setattr(
        "app.main.search_documents",
        failing_search_documents,
    )

    response = client.get(
        "/search",
        params={
            "query": "How do I create a new branch in Git?",
            "top_k": 5,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == "Internal server error"


def test_chat_handles_llm_failure(monkeypatch):

    def failing_answer_question(
        question,
        top_k=5,
        filename=None,
        file_id=None,
    ):
        raise RuntimeError("Groq API failed")

    monkeypatch.setattr(
        "app.main.answer_question",
        failing_answer_question,
    )

    response = client.get(
        "/chat",
        params={
            "query": "How do I create a new branch in Git?",
            "top_k": 5,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == "Internal server error"
