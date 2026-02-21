"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod

from rtfm.models.schemas import Chunk, QueryResult


class VectorStore(ABC):
    """Base class for vector stores."""

    @abstractmethod
    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]], collection_name: str
    ) -> int:
        """Upsert chunks with their embeddings. Returns number of chunks upserted."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        collection_name: str = "engineering",
    ) -> list[QueryResult]:
        """Query the vector store. Returns ranked results."""

    @abstractmethod
    def delete_by_source(self, source_file: str, collection_name: str) -> int:
        """Delete all chunks from a source file. Returns number deleted."""

    @abstractmethod
    def list_sources(self, collection_name: str) -> list[str]:
        """List all unique source files in a collection."""

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Count total chunks in a collection."""
