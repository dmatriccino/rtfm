"""Tests for the EPUB parser using a minimal test EPUB."""

from pathlib import Path

from ebooklib import epub

from rtfm.ingestion.epub_parser import EpubParser
from rtfm.models.schemas import ContentType


def _create_test_epub(path: Path) -> Path:
    """Create a minimal EPUB with heading, prose, and code."""
    book = epub.EpubBook()
    book.set_identifier("test-book-001")
    book.set_title("Test Engineering Book")
    book.set_language("en")
    book.add_author("Test Author")

    # Create a chapter with mixed content
    chapter = epub.EpubHtml(title="Chapter 1", file_name="chap01.xhtml", lang="en")
    chapter.content = """
    <html><body>
        <h1>Chapter 1: Getting Started</h1>
        <p>This is an introduction to software engineering principles.</p>
        <p>We will cover design patterns and best practices.</p>
        <h2>Code Examples</h2>
        <pre><code>def strategy_pattern():
    return "polymorphism"</code></pre>
        <p>The strategy pattern is useful for swapping algorithms.</p>
    </body></html>
    """
    book.add_item(chapter)

    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book)
    return path


def test_can_parse():
    parser = EpubParser()
    assert parser.can_parse(Path("test.epub"))
    assert parser.can_parse(Path("test.EPUB"))
    assert not parser.can_parse(Path("test.pdf"))


def test_parse_epub(tmp_path):
    """Parse a programmatically created EPUB."""
    epub_path = _create_test_epub(tmp_path / "test.epub")
    parser = EpubParser()

    metadata, sections = parser.parse(epub_path)

    assert metadata.title == "Test Engineering Book"
    assert metadata.file_type == "epub"
    assert len(sections) > 0


def test_heading_extraction(tmp_path):
    """Headings should be extracted with correct levels."""
    epub_path = _create_test_epub(tmp_path / "test.epub")
    parser = EpubParser()

    _, sections = parser.parse(epub_path)

    headings = [s for s in sections if s.content_type == ContentType.HEADING]
    assert len(headings) >= 1
    h1s = [h for h in headings if h.heading_level == 1]
    assert len(h1s) >= 1
    assert any("Getting Started" in h.content for h in h1s)


def test_code_extraction(tmp_path):
    """Code blocks should be extracted as CODE content type."""
    epub_path = _create_test_epub(tmp_path / "test.epub")
    parser = EpubParser()

    _, sections = parser.parse(epub_path)

    code_sections = [s for s in sections if s.content_type == ContentType.CODE]
    assert len(code_sections) >= 1
    assert any("strategy_pattern" in c.content for c in code_sections)


def test_prose_extraction(tmp_path):
    """Prose paragraphs should be extracted."""
    epub_path = _create_test_epub(tmp_path / "test.epub")
    parser = EpubParser()

    _, sections = parser.parse(epub_path)

    prose_sections = [s for s in sections if s.content_type == ContentType.PROSE]
    assert len(prose_sections) >= 1
    assert any("introduction" in p.content.lower() for p in prose_sections)


def test_heading_tracking(tmp_path):
    """Sections after a heading should reference that heading."""
    epub_path = _create_test_epub(tmp_path / "test.epub")
    parser = EpubParser()

    _, sections = parser.parse(epub_path)

    # Find prose after "Code Examples" heading
    code_examples_found = False
    for section in sections:
        if section.content_type == ContentType.HEADING and "Code Examples" in section.content:
            code_examples_found = True
        elif code_examples_found and section.content_type in (ContentType.PROSE, ContentType.CODE):
            assert section.heading == "Code Examples"
            break
