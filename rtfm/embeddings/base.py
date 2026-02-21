"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
