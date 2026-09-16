from pydantic import BaseModel


class RootResponse(BaseModel):
    message: str


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    pages: int
    chunks_indexed: int


class DocumentResponse(BaseModel):
    file_id: str
    filename: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    message: str
    file_id: str


class SearchResult(BaseModel):
    qdrant_score: float
    rerank_score: float
    filename: str
    page_start: int
    page_end: int
    text: str


class SearchResponse(BaseModel):
    query: str
    filename_filter: str | None
    file_id_filter: str | None
    results: list[SearchResult]


class ChatSource(BaseModel):
    source: int
    filename: str
    page_start: int
    page_end: int
    text: str
    qdrant_score: float
    rerank_score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[ChatSource]


# from pydantic import BaseModel


# class RootResponse(BaseModel):
#     message: str


# class UploadResponse(BaseModel):
#     file_id: str
#     filename: str
#     pages: int
#     chunks_indexed: int


# class DocumentResponse(BaseModel):
#     file_id: str
#     filename: str


# class DocumentListResponse(BaseModel):
#     documents: list[DocumentResponse]


# class SearchResult(BaseModel):
#     qdrant_score: float
#     rerank_score: float
#     filename: str
#     page_start: int
#     page_end: int
#     text: str


# class SearchResponse(BaseModel):
#     query: str
#     filename_filter: str | None
#     file_id_filter: str | None
#     results: list[SearchResult]


# class ChatSource(BaseModel):
#     source: int
#     filename: str
#     page_start: int
#     page_end: int
#     text: str
#     qdrant_score: float
#     rerank_score: float


# class ChatResponse(BaseModel):
#     query: str
#     answer: str
#     sources: list[ChatSource]


# # from pydantic import BaseModel


# # class RootResponse(BaseModel):
# #     message: str


# # class DocumentResponse(BaseModel):
# #     file_id: str
# #     filename: str


# # class DocumentListResponse(BaseModel):
# #     documents: list[DocumentResponse]


# # class UploadResponse(BaseModel):
# #     file_id: str
# #     filename: str
# #     pages: int
# #     chunks_indexed: int


# # class DeleteResponse(BaseModel):
# #     message: str
# #     file_id: str


# # class SearchResult(BaseModel):
# #     qdrant_score: float
# #     rerank_score: float
# #     filename: str
# #     page_start: int
# #     page_end: int
# #     text: str


# # class SearchResponse(BaseModel):
# #     query: str
# #     filename_filter: str | None
# #     file_id_filter: str | None
# #     results: list[SearchResult]


# # class ChatResponse(BaseModel):
# #     query: str
# #     answer: str
# #     sources: list[dict]
