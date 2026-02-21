"""Pydantic models for the RTFM system."""

import hashlib
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class ContentType(str, Enum):
    """Type of content in a section or chunk."""

    PROSE = "prose"
    CODE = "code"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"


class Section(BaseModel):
    """Raw parsed section from a document parser."""

    content: str
    content_type: ContentType = ContentType.PROSE
    heading: str | None = None
    heading_level: int | None = None
    page_number: int | None = None
    source_file: str = ""
    font_name: str | None = None
    font_size: float | None = None


class Chunk(BaseModel):
    """Processed chunk ready for embedding and storage."""

    id: str = ""
    content: str
    content_type: ContentType = ContentType.PROSE
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        if not self.id:
            source_file = self.metadata.get("source_file", "")
            hash_input = f"{source_file}::{self.content}"
            self.id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class BookMetadata(BaseModel):
    """Metadata about an ingested book."""

    title: str
    file_path: str
    file_type: str
    total_pages: int = 0
    total_chunks: int = 0
    ingested_at: datetime = Field(default_factory=datetime.now)


class QueryResult(BaseModel):
    """A single search result."""

    content: str
    score: float
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response containing multiple search results."""

    query: str
    results: list[QueryResult] = Field(default_factory=list)

    @computed_field
    @property
    def total_results(self) -> int:
        return len(self.results)
