"""EPUB document parser using ebooklib and BeautifulSoup."""

from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from rtfm.ingestion.base import DocumentParser
from rtfm.models.schemas import BookMetadata, ContentType, Section

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class EpubParser(DocumentParser):
    """Parse EPUB documents using ebooklib and BeautifulSoup."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".epub"

    def parse(self, file_path: Path) -> tuple[BookMetadata, list[Section]]:
        book = epub.read_epub(str(file_path), options={"ignore_ncx": True})

        # Extract metadata
        title_meta = book.get_metadata("DC", "title")
        title = title_meta[0][0] if title_meta else file_path.stem

        sections: list[Section] = []
        page_number = 1

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            soup = BeautifulSoup(content, "html.parser")

            current_heading: str | None = None
            current_heading_level: int | None = None

            for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "code",
                                          "ul", "ol", "table"]):
                if not isinstance(element, Tag):
                    continue

                text = element.get_text(separator="\n", strip=True)
                if not text:
                    continue

                tag_name = element.name

                if tag_name in HEADING_TAGS:
                    level = int(tag_name[1])
                    current_heading = text
                    current_heading_level = level
                    sections.append(
                        Section(
                            content=text,
                            content_type=ContentType.HEADING,
                            heading=text,
                            heading_level=level,
                            page_number=page_number,
                            source_file=str(file_path),
                        )
                    )
                elif tag_name in ("pre", "code"):
                    # Only treat block-level code as CODE
                    # Inline <code> inside <p> will be caught by the <p> handler
                    if tag_name == "code" and element.parent and element.parent.name != "pre":
                        continue
                    sections.append(
                        Section(
                            content=text,
                            content_type=ContentType.CODE,
                            heading=current_heading,
                            heading_level=current_heading_level,
                            page_number=page_number,
                            source_file=str(file_path),
                        )
                    )
                elif tag_name in ("ul", "ol"):
                    sections.append(
                        Section(
                            content=text,
                            content_type=ContentType.LIST,
                            heading=current_heading,
                            heading_level=current_heading_level,
                            page_number=page_number,
                            source_file=str(file_path),
                        )
                    )
                elif tag_name == "table":
                    sections.append(
                        Section(
                            content=text,
                            content_type=ContentType.TABLE,
                            heading=current_heading,
                            heading_level=current_heading_level,
                            page_number=page_number,
                            source_file=str(file_path),
                        )
                    )
                else:
                    sections.append(
                        Section(
                            content=text,
                            content_type=ContentType.PROSE,
                            heading=current_heading,
                            heading_level=current_heading_level,
                            page_number=page_number,
                            source_file=str(file_path),
                        )
                    )

            page_number += 1

        metadata = BookMetadata(
            title=title,
            file_path=str(file_path),
            file_type="epub",
            total_pages=page_number - 1,
        )
        return metadata, sections
