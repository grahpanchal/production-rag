from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

client = TestClient(app)

AUTH_HEADERS = {
    "X-API-Key": settings.api_key,
}


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Production RAG API is running"


def test_search_requires_query():
    response = client.get(
        "/search",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_search_rejects_empty_query():
    response = client.get(
        "/search",
        params={"query": ""},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_search_rejects_invalid_top_k():
    response = client.get(
        "/search",
        params={
            "query": "test",
            "top_k": 0,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_search_rejects_top_k_above_limit():
    response = client.get(
        "/search",
        params={
            "query": "test",
            "top_k": 21,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_requires_query():
    response = client.get(
        "/chat",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_rejects_empty_query():
    response = client.get(
        "/chat",
        params={"query": ""},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_rejects_invalid_top_k():
    response = client.get(
        "/chat",
        params={
            "query": "test",
            "top_k": 0,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_rejects_top_k_above_limit():
    response = client.get(
        "/chat",
        params={
            "query": "test",
            "top_k": 21,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "test.txt",
                b"this is not a pdf",
                "text/plain",
            )
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 415

    data = response.json()

    assert data["detail"] == "Only PDF files are supported"


def test_delete_nonexistent_document():
    fake_file_id = "00000000-0000-0000-0000-000000000000"

    response = client.delete(
        f"/documents/{fake_file_id}",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Document not found"


def test_search_rejects_missing_api_key():
    response = client.get(
        "/search",
        params={
            "query": "test",
        },
    )

    assert response.status_code == 401

    data = response.json()

    assert data["detail"] == "Invalid or missing API key"


def test_search_rejects_invalid_api_key():
    response = client.get(
        "/search",
        params={
            "query": "test",
        },
        headers={
            "X-API-Key": "wrong-api-key",
        },
    )

    assert response.status_code == 401

    data = response.json()

    assert data["detail"] == "Invalid or missing API key"


def test_search_accepts_valid_api_key():
    response = client.get(
        "/search",
        params={
            "query": "test",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200


def test_security_header_content_type_options():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_security_header_frame_options():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"


def test_security_header_referrer_policy():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["Referrer-Policy"] == "no-referrer"
