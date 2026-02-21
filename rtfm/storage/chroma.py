"""ChromaDB vector store implementation."""

import chromadb

from config.settings import Settings, get_settings
from rtfm.models.schemas import Chunk, QueryResult
from rtfm.storage.base import VectorStore

BATCH_SIZE = 5000


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB with persistent storage."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = chromadb.PersistentClient(path=str(self.settings.chroma_db_dir))

    def _get_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]], collection_name: str
    ) -> int:
        collection = self._get_collection(collection_name)

        # Upsert in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            batch_embeddings = embeddings[i : i + BATCH_SIZE]

            ids = [c.id for c in batch_chunks]
            documents = [c.content for c in batch_chunks]
            metadatas = [_sanitize_metadata(c.metadata) for c in batch_chunks]

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
        collection_name: str = "engineering",
    ) -> list[QueryResult]:
        collection = self._get_collection(collection_name)

        if collection.count() == 0:
            return []

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        query_results = []
        if results["documents"] and results["distances"]:
            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
            ):
                score = 1.0 - distance
                query_results.append(
                    QueryResult(content=doc, score=score, metadata=metadata or {})
                )

        return query_results

    def delete_by_source(self, source_file: str, collection_name: str) -> int:
        collection = self._get_collection(collection_name)

        results = collection.get(where={"source_file": source_file})
        if not results["ids"]:
            return 0

        count = len(results["ids"])
        collection.delete(ids=results["ids"])
        return count

    def list_sources(self, collection_name: str) -> list[str]:
        collection = self._get_collection(collection_name)
        results = collection.get(include=["metadatas"])

        sources = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "source_file" in meta:
                    sources.add(meta["source_file"])

        return sorted(sources)

    def count(self, collection_name: str) -> int:
        collection = self._get_collection(collection_name)
        return collection.count()


def _sanitize_metadata(metadata: dict) -> dict:
    """Ensure all metadata values are ChromaDB-compatible types."""
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
