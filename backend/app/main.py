from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Depends,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from qdrant_client.models import Filter, FieldCondition, MatchValue

import logging
import os
import shutil
import uuid
import time

from app.rag.loader import extract_text_from_pdf
from app.rag.indexer import index_document
from app.db.qdrant import client, create_collection, delete_document_chunks
from app.rag.retriever import search_documents
from app.rag.pipeline import answer_question
from app.config import settings
from app.schemas import (
    RootResponse,
    UploadResponse,
    DocumentListResponse,
    DeleteResponse,
    SearchResponse,
    ChatResponse,
)
from app.auth import verify_api_key

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_collection()
    yield


app = FastAPI(
    title="Production RAG API",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"

    return response

# app = FastAPI(title="Production RAG API")


# --------------------------------------------------
# Request latency middleware
# --------------------------------------------------


@app.middleware("http")
async def log_request_time(request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = f"{duration:.3f}"

    logger.info(
        "Request completed: method=%s path=%s status=%s duration=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


# --------------------------------------------------
# Global exception handler
# --------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(
        "Unhandled exception: method=%s path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# Startup
# --------------------------------------------------


# @app.on_event("startup")
# def startup():
#     create_collection()


# --------------------------------------------------
# Root
# --------------------------------------------------


@app.get("/", response_model=RootResponse)
def root():
    return {"message": "Production RAG API is running"}


# --------------------------------------------------
# List documents
# --------------------------------------------------


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(
    _: str = Depends(verify_api_key),
):

    documents = {}

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            payload = point.payload

            file_id = payload.get("file_id")

            if file_id and file_id not in documents:

                documents[file_id] = {
                    "file_id": file_id,
                    "filename": payload.get("filename"),
                }

        if next_offset is None:
            break

        offset = next_offset

    return {"documents": list(documents.values())}


# --------------------------------------------------
# Upload document
# --------------------------------------------------


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):

    # Validate file type

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        logger.warning(
            "Rejected upload: unsupported file type filename=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported",
        )

    # Generate unique file ID

    file_id = str(uuid.uuid4())

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{file_id}.pdf",
    )

    logger.info(
        "Document upload started: file_id=%s filename=%s",
        file_id,
        file.filename,
    )

    try:

        total_size = 0

        with open(file_path, "wb") as buffer:

            while chunk := await file.read(1024 * 1024):

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:

                    logger.warning(
                        "Rejected upload: file too large filename=%s size=%s",
                        file.filename,
                        total_size,
                    )

                    raise HTTPException(
                        status_code=413,
                        detail="File size must be 10 MB or less",
                    )

                buffer.write(chunk)

        # Extract PDF text

        pages = extract_text_from_pdf(file_path)

        # Create embeddings and index chunks

        chunks_indexed = index_document(
            file_id=file_id,
            filename=file.filename,
            pages=pages,
        )

        logger.info(
            "Document indexed successfully: file_id=%s pages=%s chunks=%s",
            file_id,
            len(pages),
            chunks_indexed,
        )

        return {
            "file_id": file_id,
            "filename": file.filename,
            "pages": len(pages),
            "chunks_indexed": chunks_indexed,
        }

    except HTTPException:

        if os.path.exists(file_path):
            os.remove(file_path)

        raise

    except Exception:

        logger.exception(
            "Document processing failed: file_id=%s filename=%s",
            file_id,
            file.filename,
        )

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail="Failed to process document",
        )


# --------------------------------------------------
# Delete document
# --------------------------------------------------


@app.delete(
    "/documents/{file_id}",
    response_model=DeleteResponse,
)
def delete_document(
    file_id: str,
    _: str = Depends(verify_api_key),
):

    logger.info(
        "Document deletion requested: file_id=%s",
        file_id,
    )

    # Check whether document exists

    existing_points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="file_id",
                    match=MatchValue(value=file_id),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    if not existing_points:

        logger.warning(
            "Document deletion failed: not found file_id=%s",
            file_id,
        )

        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    # Delete Qdrant chunks

    delete_document_chunks(file_id)

    # Delete physical PDF

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{file_id}.pdf",
    )

    if os.path.exists(file_path):
        os.remove(file_path)

    logger.info(
        "Document deleted successfully: file_id=%s",
        file_id,
    )

    return {
        "message": "Document deleted successfully",
        "file_id": file_id,
    }


# --------------------------------------------------
# Search
# --------------------------------------------------


@app.get(
    "/search",
    response_model=SearchResponse,
)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    filename: str | None = None,
    file_id: str | None = None,
    _: str = Depends(verify_api_key),
):

    results = search_documents(
        query=query,
        top_k=top_k,
        filename=filename,
        file_id=file_id,
    )

    return {
        "query": query,
        "filename_filter": filename,
        "file_id_filter": file_id,
        "results": [
            {
                "qdrant_score": result["document"].score,
                "rerank_score": result["rerank_score"],
                "filename": result["document"].payload["filename"],
                "page_start": result["document"].payload["page_start"],
                "page_end": result["document"].payload["page_end"],
                "text": result["document"].payload["text"],
            }
            for result in results
        ],
    }


# --------------------------------------------------
# Chat
# --------------------------------------------------


@app.get(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    filename: str | None = None,
    file_id: str | None = None,
    _: str = Depends(verify_api_key),
):

    result = answer_question(
        question=query,
        top_k=top_k,
        filename=filename,
        file_id=file_id,
    )

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"],
    }


# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# from fastapi.responses import JSONResponse
# from qdrant_client.models import Filter, FieldCondition, MatchValue

# import logging
# import os
# import shutil
# import uuid

# from app.rag.loader import extract_text_from_pdf
# from app.rag.indexer import index_document
# from app.db.qdrant import client, create_collection, delete_document_chunks
# from app.rag.retriever import search_documents
# from app.rag.pipeline import answer_question
# from app.config import settings
# from app.schemas import (
#     RootResponse,
#     UploadResponse,
#     DocumentListResponse,
#     DeleteResponse,
#     SearchResponse,
#     ChatResponse,
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )

# logger = logging.getLogger(__name__)

# app = FastAPI(title="Production RAG API")

# MAX_FILE_SIZE = 50 * 1024 * 1024  # 10 MB


# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):

#     logger.exception(
#         "Unhandled exception: method=%s path=%s",
#         request.method,
#         request.url.path,
#     )

#     return JSONResponse(
#         status_code=500,
#         content={
#             "detail": "Internal server error",
#         },
#     )


# UPLOAD_DIR = "uploads"

# os.makedirs(UPLOAD_DIR, exist_ok=True)


# @app.on_event("startup")
# def startup():

#     create_collection()


# @app.get("/", response_model=RootResponse)
# def root():
#     return {"message": "Production RAG API is running"}


# @app.get("/documents", response_model=DocumentListResponse)
# def list_documents():

#     documents = {}

#     offset = None

#     while True:

#         points, next_offset = client.scroll(
#             collection_name=settings.qdrant_collection,
#             limit=100,
#             offset=offset,
#             with_payload=True,
#             with_vectors=False,
#         )

#         for point in points:

#             payload = point.payload

#             file_id = payload.get("file_id")

#             if file_id and file_id not in documents:

#                 documents[file_id] = {
#                     "file_id": file_id,
#                     "filename": payload.get("filename"),
#                 }

#         if next_offset is None:
#             break

#         offset = next_offset

#     return {"documents": list(documents.values())}


# @app.post("/documents/upload", response_model=UploadResponse)
# async def upload_document(file: UploadFile = File(...)):

#     if not file.filename or not file.filename.lower().endswith(".pdf"):
#         logger.warning(
#             "Rejected upload: unsupported file type filename=%s",
#             file.filename,
#         )

#         raise HTTPException(
#             status_code=415,
#             detail="Only PDF files are supported",
#         )

#     file_id = str(uuid.uuid4())
#     file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

#     logger.info(
#         "Document upload started: file_id=%s filename=%s",
#         file_id,
#         file.filename,
#     )

#     try:
#         total_size = 0

#         with open(file_path, "wb") as buffer:

#             while chunk := await file.read(1024 * 1024):

#                 total_size += len(chunk)

#                 if total_size > MAX_FILE_SIZE:
#                     logger.warning(
#                         "Rejected upload: file too large filename=%s size=%s",
#                         file.filename,
#                         total_size,
#                     )

#                     raise HTTPException(
#                         status_code=413,
#                         detail="File size must be 10 MB or less",
#                     )

#                 buffer.write(chunk)

#         pages = extract_text_from_pdf(file_path)

#         chunks_indexed = index_document(
#             file_id=file_id,
#             filename=file.filename,
#             pages=pages,
#         )

#         logger.info(
#             "Document indexed successfully: file_id=%s pages=%s chunks=%s",
#             file_id,
#             len(pages),
#             chunks_indexed,
#         )

#         return {
#             "file_id": file_id,
#             "filename": file.filename,
#             "pages": len(pages),
#             "chunks_indexed": chunks_indexed,
#         }

#     except HTTPException:
#         if os.path.exists(file_path):
#             os.remove(file_path)

#         raise

#     except Exception:
#         logger.exception(
#             "Document processing failed: file_id=%s filename=%s",
#             file_id,
#             file.filename,
#         )

#         if os.path.exists(file_path):
#             os.remove(file_path)

#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process document",
#         )


# @app.delete(
#     "/documents/{file_id}",
#     response_model=DeleteResponse,
# )
# def delete_document(file_id: str):

#     logger.info(
#         "Document deletion requested: file_id=%s",
#         file_id,
#     )

#     existing_points, _ = client.scroll(
#         collection_name=settings.qdrant_collection,
#         scroll_filter=Filter(
#             must=[
#                 FieldCondition(
#                     key="file_id",
#                     match=MatchValue(value=file_id),
#                 )
#             ]
#         ),
#         limit=1,
#         with_payload=False,
#         with_vectors=False,
#     )

#     if not existing_points:

#         logger.warning(
#             "Document deletion failed: not found file_id=%s",
#             file_id,
#         )

#         raise HTTPException(
#             status_code=404,
#             detail="Document not found",
#         )

#     delete_document_chunks(file_id)

#     file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

#     if os.path.exists(file_path):
#         os.remove(file_path)

#     logger.info(
#         "Document deleted successfully: file_id=%s",
#         file_id,
#     )

#     return {
#         "message": "Document deleted successfully",
#         "file_id": file_id,
#     }


# @app.get(
#     "/search",
#     response_model=SearchResponse,
# )
# def search(
#     query: str = Query(..., min_length=1),
#     top_k: int = Query(5, ge=1, le=20),
#     filename: str | None = None,
#     file_id: str | None = None,
# ):

#     results = search_documents(
#         query=query,
#         top_k=top_k,
#         filename=filename,
#         file_id=file_id,
#     )

#     return {
#         "query": query,
#         "filename_filter": filename,
#         "file_id_filter": file_id,
#         "results": [
#             {
#                 "qdrant_score": result["document"].score,
#                 "rerank_score": result["rerank_score"],
#                 "filename": result["document"].payload["filename"],
#                 "page_start": result["document"].payload["page_start"],
#                 "page_end": result["document"].payload["page_end"],
#                 "text": result["document"].payload["text"],
#             }
#             for result in results
#         ],
#     }


# @app.get(
#     "/chat",
#     response_model=ChatResponse,
# )
# def chat(
#     query: str = Query(..., min_length=1),
#     top_k: int = Query(5, ge=1, le=20),
#     filename: str | None = None,
#     file_id: str | None = None,
# ):

#     result = answer_question(
#         question=query,
#         top_k=top_k,
#         filename=filename,
#         file_id=file_id,
#     )

#     return {
#         "query": query,
#         "answer": result["answer"],
#         "sources": result["sources"],
#     }
