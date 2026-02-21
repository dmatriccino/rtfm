"""Ingestion pipeline that orchestrates parsing, chunking, embedding, and storage."""

from pathlib import Path

import structlog

from config.settings import Settings, get_settings
from rtfm.embeddings.base import EmbeddingProvider
from rtfm.ingestion.base import DocumentParser
from rtfm.ingestion.chunker import SemanticChunker
from rtfm.storage.base import VectorStore

logger = structlog.get_logger()

EMBED_BATCH_SIZE = 256


class IngestionPipeline:
    """Orchestrate document ingestion: parse → chunk → embed → store."""

    def __init__(
        self,
        parsers: list[DocumentParser],
        chunker: SemanticChunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ):
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings or get_settings()

        # Ensure data directories exist
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.books_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(
        self,
        file_path: Path,
        collection: str | None = None,
        force: bool = False,
    ) -> int:
        """Ingest a single file. Returns number of chunks stored."""
        collection = collection or self.settings.default_collection
        file_path = Path(file_path).resolve()

        parser = self._find_parser(file_path)
        if parser is None:
            logger.warning("no_parser_found", file=str(file_path))
            return 0

        log = logger.bind(file=str(file_path), collection=collection)

        if force:
            deleted = self.vector_store.delete_by_source(str(file_path), collection)
            log.info("deleted_existing_chunks", count=deleted)

        # Parse
        log.info("parsing")
        metadata, sections = parser.parse(file_path)
        log.info("parsed", sections=len(sections), pages=metadata.total_pages)

        # Chunk
        chunks = self.chunker.chunk(sections, metadata)
        log.info("chunked", chunks=len(chunks))

        if not chunks:
            return 0

        # Embed in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = self.embedder.embed_texts(texts)
            all_embeddings.extend(embeddings)
            log.info("embedded_batch", batch=i // EMBED_BATCH_SIZE + 1)

        # Store
        count = self.vector_store.upsert_chunks(chunks, all_embeddings, collection)
        log.info("stored", chunks=count)

        return count

    def ingest_directory(
        self,
        dir_path: Path,
        collection: str | None = None,
        force: bool = False,
    ) -> int:
        """Ingest all supported files in a directory. Returns total chunks stored."""
        dir_path = Path(dir_path).resolve()
        total = 0

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and self._find_parser(file_path) is not None:
                count = self.ingest_file(file_path, collection, force)
                total += count

        return total

    def _find_parser(self, file_path: Path) -> DocumentParser | None:
        """Find a parser that can handle the given file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
