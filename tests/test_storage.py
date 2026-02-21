"""Tests for the ChromaDB vector store."""

import pytest

from config.settings import Settings
from rtfm.models.schemas import Chunk, ContentType
from rtfm.storage.chroma import ChromaVectorStore


@pytest.fixture
def store(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        books_dir=tmp_path / "books",
        chroma_db_dir=tmp_path / "chroma_db",
    )
    return ChromaVectorStore(settings)


def _make_chunks(n=3, source="test.pdf"):
    return [
        Chunk(
            content=f"chunk content {i}",
            content_type=ContentType.PROSE,
            metadata={"source_file": source, "book_title": "Test Book", "content_type": "prose"},
        )
        for i in range(n)
    ]


def _make_embeddings(n=3, dim=384):
    return [[float(i + j) / 100 for j in range(dim)] for i in range(n)]


def test_upsert_and_count(store):
    chunks = _make_chunks(3)
    embeddings = _make_embeddings(3)
    count = store.upsert_chunks(chunks, embeddings, "test")
    assert count == 3
    assert store.count("test") == 3


def test_query(store):
    chunks = _make_chunks(3)
    embeddings = _make_embeddings(3)
    store.upsert_chunks(chunks, embeddings, "test")

    results = store.query(embeddings[0], top_k=2, collection_name="test")
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_delete_by_source(store):
    chunks = _make_chunks(3, source="a.pdf") + _make_chunks(2, source="b.pdf")
    embeddings = _make_embeddings(5)
    store.upsert_chunks(chunks, embeddings, "test")

    assert store.count("test") == 5
    deleted = store.delete_by_source("a.pdf", "test")
    assert deleted == 3
    assert store.count("test") == 2


def test_list_sources(store):
    chunks = _make_chunks(2, source="a.pdf") + _make_chunks(2, source="b.pdf")
    embeddings = _make_embeddings(4)
    store.upsert_chunks(chunks, embeddings, "test")

    sources = store.list_sources("test")
    assert sorted(sources) == ["a.pdf", "b.pdf"]


def test_idempotent_upsert(store):
    """Upserting same chunks twice should not duplicate."""
    chunks = _make_chunks(3)
    embeddings = _make_embeddings(3)
    store.upsert_chunks(chunks, embeddings, "test")
    store.upsert_chunks(chunks, embeddings, "test")
    assert store.count("test") == 3


def test_empty_collection(store):
    assert store.count("nonexistent") == 0
    assert store.list_sources("nonexistent") == []
    results = store.query([0.0] * 384, top_k=5, collection_name="nonexistent")
    assert results == []
