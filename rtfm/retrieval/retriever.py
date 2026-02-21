"""Retriever that combines embedding and vector search."""

from config.settings import Settings, get_settings
from rtfm.embeddings.base import EmbeddingProvider
from rtfm.models.schemas import QueryResponse, QueryResult
from rtfm.storage.base import VectorStore


class Retriever:
    """Embed queries and search the vector store for relevant chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ):
        self.embedder = embedding_provider
        self.store = vector_store
        self.settings = settings or get_settings()

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        book_title: str | None = None,
        content_type: str | None = None,
        collection_name: str | None = None,
    ) -> QueryResponse:
        """Search for relevant chunks matching the query."""
        top_k = top_k or self.settings.default_top_k
        collection_name = collection_name or self.settings.default_collection

        query_embedding = self.embedder.embed_query(query_text)

        where = self._build_filter(book_title, content_type)

        results: list[QueryResult] = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
            collection_name=collection_name,
        )

        return QueryResponse(query=query_text, results=results)

    def _build_filter(
        self, book_title: str | None, content_type: str | None
    ) -> dict | None:
        """Build ChromaDB where filter from optional parameters."""
        conditions = []

        if book_title:
            conditions.append({"book_title": book_title})
        if content_type:
            conditions.append({"content_type": content_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
