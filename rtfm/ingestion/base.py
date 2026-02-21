"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod
from pathlib import Path

from rtfm.models.schemas import BookMetadata, Section


class DocumentParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""

    @abstractmethod
    def parse(self, file_path: Path) -> tuple[BookMetadata, list[Section]]:
        """Parse a document into metadata and sections."""
