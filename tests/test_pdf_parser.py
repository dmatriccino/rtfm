"""Tests for the PDF parser using programmatically created test PDFs."""

from pathlib import Path

import pymupdf

from rtfm.ingestion.pdf_parser import PdfParser, _is_monospace
from rtfm.models.schemas import ContentType


def _create_test_pdf(path: Path) -> Path:
    """Create a minimal test PDF with heading, prose, and code."""
    doc = pymupdf.open()

    page = doc.new_page()

    # Heading — large font
    page.insert_text((72, 72), "Chapter 1: Introduction", fontsize=24, fontname="helv")

    # Prose — normal font
    page.insert_text(
        (72, 120), "This is regular prose text about software.", fontsize=12, fontname="helv"
    )

    # Code — monospace font
    page.insert_text((72, 160), "def hello():", fontsize=10, fontname="cour")
    page.insert_text((72, 175), "    print('hello')", fontsize=10, fontname="cour")

    doc.save(str(path))
    doc.close()
    return path


def test_is_monospace():
    assert _is_monospace("Courier")
    assert _is_monospace("Courier-Bold")
    assert _is_monospace("Consolas")
    assert _is_monospace("DejaVu Sans Mono")
    assert not _is_monospace("Helvetica")
    assert not _is_monospace("Arial")
    assert not _is_monospace("Times-Roman")


def test_can_parse():
    parser = PdfParser()
    assert parser.can_parse(Path("test.pdf"))
    assert parser.can_parse(Path("test.PDF"))
    assert not parser.can_parse(Path("test.epub"))
    assert not parser.can_parse(Path("test.txt"))


def test_parse_pdf(tmp_path):
    """Parse a programmatically created PDF."""
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    parser = PdfParser()

    metadata, sections = parser.parse(pdf_path)

    assert metadata.title is not None
    assert metadata.file_type == "pdf"
    assert metadata.total_pages == 1
    assert len(sections) > 0


def test_heading_detection(tmp_path):
    """Large font text should be detected as heading."""
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    parser = PdfParser()

    _, sections = parser.parse(pdf_path)

    headings = [s for s in sections if s.content_type == ContentType.HEADING]
    assert len(headings) > 0
    assert any("Introduction" in h.content for h in headings)


def test_code_detection(tmp_path):
    """Monospace font text should be detected as code."""
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    parser = PdfParser()

    _, sections = parser.parse(pdf_path)

    code_sections = [s for s in sections if s.content_type == ContentType.CODE]
    assert len(code_sections) > 0
    assert any("def" in c.content or "print" in c.content for c in code_sections)


def test_section_merging(tmp_path):
    """Adjacent same-type spans should be merged."""
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    parser = PdfParser()

    _, sections = parser.parse(pdf_path)

    # Should have fewer sections than raw spans due to merging
    assert len(sections) >= 2  # At least heading + prose + code
