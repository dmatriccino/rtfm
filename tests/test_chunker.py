"""Tests for the semantic chunker."""

from config.settings import Settings
from rtfm.ingestion.chunker import SemanticChunker
from rtfm.models.schemas import BookMetadata, ContentType, Section


def _settings(**kwargs):
    return Settings(
        data_dir="/tmp/test",
        books_dir="/tmp/test/books",
        chroma_db_dir="/tmp/test/chroma",
        **kwargs,
    )


def _metadata():
    return BookMetadata(title="Test", file_path="/tmp/test.pdf", file_type="pdf", total_pages=1)


def test_prose_splitting():
    """Prose exceeding max size should be split."""
    settings = _settings(max_chunk_size=50, chunk_overlap=0)
    chunker = SemanticChunker(settings)
    meta = _metadata()

    long_text = "This is a sentence. " * 20  # ~400 chars
    sections = [
        Section(content=long_text, content_type=ContentType.PROSE, source_file="/tmp/test.pdf")
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content_type == ContentType.PROSE


def test_code_preservation():
    """Short code blocks should be kept intact."""
    settings = _settings(max_code_chunk_size=3000)
    chunker = SemanticChunker(settings)
    meta = _metadata()

    code = "def foo():\n    return 42"
    sections = [
        Section(content=code, content_type=ContentType.CODE, source_file="/tmp/test.pdf")
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) == 1
    assert chunks[0].content == code
    assert chunks[0].content_type == ContentType.CODE


def test_heading_propagation():
    """Headings should be captured in subsequent chunk metadata."""
    settings = _settings()
    chunker = SemanticChunker(settings)
    meta = _metadata()

    sections = [
        Section(
            content="My Heading",
            content_type=ContentType.HEADING,
            heading="My Heading",
            source_file="/tmp/test.pdf",
        ),
        Section(
            content="Some prose text under the heading.",
            content_type=ContentType.PROSE,
            source_file="/tmp/test.pdf",
        ),
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading"] == "My Heading"


def test_heading_not_standalone_chunk():
    """Headings should not produce their own chunks."""
    settings = _settings()
    chunker = SemanticChunker(settings)
    meta = _metadata()

    sections = [
        Section(
            content="Just a Heading",
            content_type=ContentType.HEADING,
            source_file="/tmp/test.pdf",
        ),
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) == 0


def test_overlap():
    """Consecutive prose chunks should have overlap text."""
    settings = _settings(max_chunk_size=50, chunk_overlap=20)
    chunker = SemanticChunker(settings)
    meta = _metadata()

    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    sections = [
        Section(content=text, content_type=ContentType.PROSE, source_file="/tmp/test.pdf")
    ]

    chunks = chunker.chunk(sections, meta)
    if len(chunks) > 1:
        # Second chunk should contain some text from end of first
        first_end = chunks[0].content[-15:]
        assert any(word in chunks[1].content for word in first_end.split())


def test_large_code_splitting():
    """Large code blocks should be split at blank lines."""
    settings = _settings(max_code_chunk_size=100)
    chunker = SemanticChunker(settings)
    meta = _metadata()

    code = (
        "def foo():\n    pass\n\ndef bar():\n    pass\n\n"
        "def baz():\n    pass\n\ndef qux():\n    return 1\n\n"
        "def quux():\n    return 2"
    )
    sections = [
        Section(content=code, content_type=ContentType.CODE, source_file="/tmp/test.pdf")
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content_type == ContentType.CODE


def test_chunk_metadata_fields():
    """Chunks should have all required metadata fields."""
    settings = _settings()
    chunker = SemanticChunker(settings)
    meta = _metadata()

    sections = [
        Section(
            content="Some text",
            content_type=ContentType.PROSE,
            heading="A Heading",
            page_number=5,
            source_file="/tmp/test.pdf",
        )
    ]

    chunks = chunker.chunk(sections, meta)
    assert len(chunks) == 1
    m = chunks[0].metadata
    assert m["book_title"] == "Test"
    assert m["source_file"] == "/tmp/test.pdf"
    assert m["page_number"] == 5
    assert m["heading"] == "A Heading"
    assert m["content_type"] == "prose"
    assert m["char_count"] == len("Some text")
