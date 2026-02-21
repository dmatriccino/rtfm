"""Shared test fixtures."""

# Monkey-patch sqlite3 for environments where _sqlite3 is missing
import sys

try:
    import sqlite3  # noqa: F401
except ModuleNotFoundError:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3

import pytest

from config.settings import Settings
from rtfm.models.schemas import BookMetadata, ContentType, Section


@pytest.fixture
def tmp_settings(tmp_path):
    """Settings using tmp_path for all data directories."""
    return Settings(
        data_dir=tmp_path / "data",
        books_dir=tmp_path / "data" / "books",
        chroma_db_dir=tmp_path / "data" / "chroma_db",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_metadata():
    """Sample book metadata."""
    return BookMetadata(
        title="Test Book",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        total_pages=10,
    )


@pytest.fixture
def sample_sections():
    """Sample parsed sections with headings, prose, and code."""
    return [
        Section(
            content="Chapter 1: Introduction",
            content_type=ContentType.HEADING,
            heading="Chapter 1: Introduction",
            heading_level=1,
            page_number=1,
            source_file="/tmp/test.pdf",
        ),
        Section(
            content=(
                "This is the introduction to the book. "
                "It covers the basics of software engineering."
            ),
            content_type=ContentType.PROSE,
            heading="Chapter 1: Introduction",
            heading_level=1,
            page_number=1,
            source_file="/tmp/test.pdf",
        ),
        Section(
            content="def hello():\n    print('hello world')",
            content_type=ContentType.CODE,
            heading="Chapter 1: Introduction",
            heading_level=1,
            page_number=2,
            source_file="/tmp/test.pdf",
        ),
        Section(
            content="Design Patterns",
            content_type=ContentType.HEADING,
            heading="Design Patterns",
            heading_level=2,
            page_number=3,
            source_file="/tmp/test.pdf",
        ),
        Section(
            content=(
                "The strategy pattern defines a family of algorithms. "
                "Each algorithm is encapsulated and made interchangeable."
            ),
            content_type=ContentType.PROSE,
            heading="Design Patterns",
            heading_level=2,
            page_number=3,
            source_file="/tmp/test.pdf",
        ),
    ]
