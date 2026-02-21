"""Tests for the retriever."""

from unittest.mock import MagicMock

from rtfm.models.schemas import QueryResult
from rtfm.retrieval.retriever import Retriever


def test_query_basic():
    """Retriever should embed query and search store."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    mock_store = MagicMock()
    mock_store.query.return_value = [
        QueryResult(content="result 1", score=0.9, metadata={"book_title": "Test"}),
        QueryResult(content="result 2", score=0.8, metadata={"book_title": "Test"}),
    ]

    retriever = Retriever(mock_embedder, mock_store)
    response = retriever.query("what is a pattern?", top_k=2, collection_name="test")

    assert response.query == "what is a pattern?"
    assert response.total_results == 2
    mock_embedder.embed_query.assert_called_once_with("what is a pattern?")


def test_filter_by_book_title():
    """Should build a where filter for book_title."""
    retriever = Retriever(MagicMock(), MagicMock())
    f = retriever._build_filter(book_title="My Book", content_type=None)
    assert f == {"book_title": "My Book"}


def test_filter_by_content_type():
    """Should build a where filter for content_type."""
    retriever = Retriever(MagicMock(), MagicMock())
    f = retriever._build_filter(book_title=None, content_type="code")
    assert f == {"content_type": "code"}


def test_filter_combined():
    """Should build an $and filter for multiple params."""
    retriever = Retriever(MagicMock(), MagicMock())
    f = retriever._build_filter(book_title="My Book", content_type="code")
    assert f == {"$and": [{"book_title": "My Book"}, {"content_type": "code"}]}


def test_filter_none():
    """No filters should return None."""
    retriever = Retriever(MagicMock(), MagicMock())
    f = retriever._build_filter(book_title=None, content_type=None)
    assert f is None
