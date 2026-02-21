"""Tests for the sentence-transformer embedding provider."""

import pytest

from config.settings import Settings
from rtfm.embeddings.local import SentenceTransformerEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance (model loads once)."""
    settings = Settings(
        data_dir="/tmp/test",
        books_dir="/tmp/test/books",
        chroma_db_dir="/tmp/test/chroma",
    )
    return SentenceTransformerEmbedder(settings)


def test_dimensions(embedder):
    assert embedder.dimensions == 384


def test_embed_query(embedder):
    result = embedder.embed_query("what is the strategy pattern?")
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_embed_texts_batch(embedder):
    texts = ["hello world", "design patterns", "python programming"]
    results = embedder.embed_texts(texts)
    assert len(results) == 3
    assert all(len(r) == 384 for r in results)


def test_deterministic_output(embedder):
    """Same input should produce same output."""
    text = "the observer pattern"
    r1 = embedder.embed_query(text)
    r2 = embedder.embed_query(text)
    assert r1 == r2
