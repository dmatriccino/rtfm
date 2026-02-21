"""Local embedding provider using sentence-transformers."""

from sentence_transformers import SentenceTransformer

from config.settings import Settings, get_settings
from rtfm.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Generate embeddings using a local sentence-transformer model."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = SentenceTransformer(self.settings.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    @property
    def dimensions(self) -> int:
        return self.settings.embedding_dimensions
