"""The API's request and response shapes.

Pydantic lives only here, at the edge: these models validate what comes in and
describe what goes out (and generate the OpenAPI page at ``/docs``). Inside,
the index keeps its Lab 2-3 dataclasses; ``app.py`` converts between the two.
"""

from typing import Literal

from pydantic import BaseModel, Field

type Status = Literal["ok"]


class Hit(BaseModel):
    rank: int = Field(ge=1)
    doc_id: int
    title: str
    score: float
    source: str = Field(description="File name, or the URL of a crawled page.")
    url: str | None = None
    snippet: str = Field(description="Text around the best match.")
    highlights: list[tuple[int, int]] = Field(
        description="[start, end) character ranges of the query terms in `snippet`."
    )


class SearchResponse(BaseModel):
    query: str
    parsed: str = Field(description="How the query was understood.")
    scorer: Literal["bm25", "tfidf"]
    page: int
    k: int
    total: int = Field(description="Documents matching the query, on all pages.")
    pages: int
    took_ms: float
    results: list[Hit]


class Document(BaseModel):
    doc_id: int
    title: str
    source: str
    url: str | None
    tokens: int
    chars: int | None = Field(description="Length of the text; null if unavailable.")
    offset: int
    text: str | None = Field(description="The requested slice of the text.")
    next_offset: int | None = None


class Stats(BaseModel):
    version: str
    documents: int
    terms: int
    tokens: int
    avg_doc_length: float
    positions: bool
    index_path: str
    index_bytes: int
    uptime_s: float


class Health(BaseModel):
    status: Status
    documents: int


class Error(BaseModel):
    detail: str
    request_id: str | None = None
