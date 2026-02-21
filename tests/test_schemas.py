"""Tests for Pydantic models."""

from rtfm.models.schemas import (
    BookMetadata,
    Chunk,
    ContentType,
    QueryResponse,
    QueryResult,
    Section,
)


def test_content_type_values():
    assert ContentType.PROSE == "prose"
    assert ContentType.CODE == "code"
    assert ContentType.HEADING == "heading"


def test_section_construction():
    s = Section(content="hello", content_type=ContentType.PROSE, source_file="test.pdf")
    assert s.content == "hello"
    assert s.content_type == ContentType.PROSE
    assert s.heading is None
    assert s.page_number is None


def test_section_with_all_fields():
    s = Section(
        content="Chapter 1",
        content_type=ContentType.HEADING,
        heading="Chapter 1",
        heading_level=1,
        page_number=1,
        source_file="book.pdf",
        font_name="Arial",
        font_size=18.0,
    )
    assert s.heading_level == 1
    assert s.font_size == 18.0


def test_chunk_id_determinism():
    """Same content + source should produce same ID."""
    c1 = Chunk(content="hello world", metadata={"source_file": "test.pdf"})
    c2 = Chunk(content="hello world", metadata={"source_file": "test.pdf"})
    assert c1.id == c2.id
    assert len(c1.id) == 16


def test_chunk_id_uniqueness():
    """Different content should produce different IDs."""
    c1 = Chunk(content="hello world", metadata={"source_file": "test.pdf"})
    c2 = Chunk(content="goodbye world", metadata={"source_file": "test.pdf"})
    assert c1.id != c2.id


def test_chunk_explicit_id():
    """Explicit ID should be used as-is."""
    c = Chunk(id="custom-id", content="test")
    assert c.id == "custom-id"


def test_book_metadata():
    m = BookMetadata(title="My Book", file_path="/tmp/book.pdf", file_type="pdf", total_pages=100)
    assert m.title == "My Book"
    assert m.total_chunks == 0
    assert m.ingested_at is not None


def test_query_result():
    r = QueryResult(content="result text", score=0.95, metadata={"book": "test"})
    assert r.score == 0.95


def test_query_response_total_results():
    response = QueryResponse(
        query="test query",
        results=[
            QueryResult(content="a", score=0.9),
            QueryResult(content="b", score=0.8),
        ],
    )
    assert response.total_results == 2


def test_query_response_empty():
    response = QueryResponse(query="empty")
    assert response.total_results == 0
