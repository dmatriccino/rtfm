"""PDF document parser using PyMuPDF."""

import statistics
from pathlib import Path

import pymupdf

from rtfm.ingestion.base import DocumentParser
from rtfm.models.schemas import BookMetadata, ContentType, Section

MONOSPACE_FONTS = {"courier", "mono", "consolas", "menlo", "firacode", "sourcecodepro", "dejavu"}


def _is_monospace(font_name: str) -> bool:
    """Check if a font name indicates a monospace/code font."""
    name = font_name.lower().replace(" ", "").replace("-", "")
    return any(mono in name for mono in MONOSPACE_FONTS)


class PdfParser(DocumentParser):
    """Parse PDF documents using PyMuPDF."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> tuple[BookMetadata, list[Section]]:
        doc = pymupdf.open(str(file_path))

        # Extract TOC for heading assignment
        toc = doc.get_toc()
        toc_headings = {entry[1].strip(): entry[0] for entry in toc}

        # First pass: collect all font sizes for median calculation
        all_font_sizes: list[float] = []
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            all_font_sizes.append(span["size"])

        median_size = statistics.median(all_font_sizes) if all_font_sizes else 12.0
        heading_threshold = median_size * 1.2

        # Second pass: extract sections
        raw_sections: list[Section] = []
        for page_num, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        font_name = span.get("font", "")
                        font_size = span.get("size", 12.0)

                        if _is_monospace(font_name):
                            content_type = ContentType.CODE
                        elif font_size >= heading_threshold:
                            content_type = ContentType.HEADING
                        else:
                            content_type = ContentType.PROSE

                        raw_sections.append(
                            Section(
                                content=text,
                                content_type=content_type,
                                page_number=page_num,
                                source_file=str(file_path),
                                font_name=font_name,
                                font_size=font_size,
                            )
                        )

        # Merge adjacent same-type spans and assign headings
        sections = _merge_sections(raw_sections, toc_headings)

        title = doc.metadata.get("title", "") or file_path.stem
        metadata = BookMetadata(
            title=title,
            file_path=str(file_path),
            file_type="pdf",
            total_pages=len(doc),
        )
        doc.close()
        return metadata, sections


def _merge_sections(
    raw_sections: list[Section], toc_headings: dict[str, int]
) -> list[Section]:
    """Merge adjacent sections of the same type and assign headings."""
    if not raw_sections:
        return []

    merged: list[Section] = []
    current = raw_sections[0].model_copy()

    for section in raw_sections[1:]:
        if (
            section.content_type == current.content_type
            and section.page_number == current.page_number
        ):
            current.content += " " + section.content
        else:
            merged.append(current)
            current = section.model_copy()
    merged.append(current)

    # Assign headings: track current heading and apply to subsequent sections
    current_heading: str | None = None
    current_heading_level: int | None = None

    for section in merged:
        if section.content_type == ContentType.HEADING:
            current_heading = section.content
            current_heading_level = toc_headings.get(section.content, 1)
            section.heading = current_heading
            section.heading_level = current_heading_level
        else:
            section.heading = current_heading
            section.heading_level = current_heading_level

    return merged
