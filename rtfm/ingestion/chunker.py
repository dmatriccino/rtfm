"""Semantic chunker that splits sections into embedding-ready chunks."""

import hashlib
import re

from config.settings import Settings, get_settings
from rtfm.models.schemas import BookMetadata, Chunk, ContentType, Section


class SemanticChunker:
    """Split parsed sections into chunks suitable for embedding."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def chunk(self, sections: list[Section], metadata: BookMetadata) -> list[Chunk]:
        """Convert sections into chunks with metadata."""
        chunks: list[Chunk] = []
        current_heading: str | None = None

        for section in sections:
            if section.content_type == ContentType.HEADING:
                current_heading = section.content
                continue

            heading = section.heading or current_heading

            if section.content_type == ContentType.CODE:
                new_chunks = self._chunk_code(section, metadata, heading)
            else:
                new_chunks = self._chunk_prose(section, metadata, heading)

            chunks.extend(new_chunks)

        return chunks

    def _chunk_prose(
        self, section: Section, metadata: BookMetadata, heading: str | None
    ) -> list[Chunk]:
        """Split prose by paragraph boundaries, then by sentence if needed."""
        text = section.content.strip()
        if not text:
            return []

        max_size = self.settings.max_chunk_size
        overlap = self.settings.chunk_overlap

        # Split by paragraph boundaries
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        raw_chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + len(para) + 2 <= max_size:
                current += "\n\n" + para
            else:
                raw_chunks.append(current)
                current = para

        if current:
            raw_chunks.append(current)

        # Split any chunks that are still too large by sentence
        split_chunks: list[str] = []
        for chunk_text in raw_chunks:
            if len(chunk_text) <= max_size:
                split_chunks.append(chunk_text)
            else:
                split_chunks.extend(self._split_by_sentence(chunk_text, max_size))

        # Apply overlap between consecutive chunks
        final_chunks = self._apply_overlap(split_chunks, overlap)

        return [
            self._make_chunk(text, section, metadata, heading)
            for text in final_chunks
            if text.strip()
        ]

    def _chunk_code(
        self, section: Section, metadata: BookMetadata, heading: str | None
    ) -> list[Chunk]:
        """Keep code blocks intact if possible; split at blank lines if too large."""
        text = section.content.strip()
        if not text:
            return []

        max_size = self.settings.max_code_chunk_size

        if len(text) <= max_size:
            return [self._make_chunk(text, section, metadata, heading)]

        # Split at blank lines
        blocks = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for block in blocks:
            if not current:
                current = block
            elif len(current) + len(block) + 2 <= max_size:
                current += "\n\n" + block
            else:
                chunks.append(current)
                current = block

        if current:
            chunks.append(current)

        # No overlap for code chunks
        return [
            self._make_chunk(t, section, metadata, heading)
            for t in chunks
            if t.strip()
        ]

    def _split_by_sentence(self, text: str, max_size: int) -> list[str]:
        """Split text by sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= max_size:
                current += " " + sentence
            else:
                chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        """Add overlap text from the end of each chunk to the beginning of the next."""
        if len(chunks) <= 1 or overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            # Find a word boundary in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            result.append(overlap_text + " " + chunks[i])

        return result

    def _make_chunk(
        self,
        content: str,
        section: Section,
        metadata: BookMetadata,
        heading: str | None,
    ) -> Chunk:
        """Create a Chunk with full metadata."""
        chunk_metadata = {
            "book_title": metadata.title,
            "source_file": str(metadata.file_path),
            "page_number": section.page_number,
            "heading": heading,
            "content_type": section.content_type.value,
            "char_count": len(content),
        }

        chunk_id = hashlib.sha256(
            f"{metadata.file_path}::{content}".encode()
        ).hexdigest()[:16]

        return Chunk(
            id=chunk_id,
            content=content,
            content_type=section.content_type,
            metadata=chunk_metadata,
        )
