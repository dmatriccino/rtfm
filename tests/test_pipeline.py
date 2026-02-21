"""Tests for the ingestion pipeline."""

from pathlib import Path

import pymupdf
import pytest
from ebooklib import epub

from config.settings import Settings
from rtfm.embeddings.local import SentenceTransformerEmbedder
from rtfm.ingestion.chunker import SemanticChunker
from rtfm.ingestion.epub_parser import EpubParser
from rtfm.ingestion.pdf_parser import PdfParser
from rtfm.ingestion.pipeline import IngestionPipeline
from rtfm.storage.chroma import ChromaVectorStore


def _create_test_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter 1: Testing", fontsize=24, fontname="helv")
    page.insert_text(
        (72, 120),
        "This is test content about software engineering patterns.",
        fontsize=12,
        fontname="helv",
    )
    page.insert_text(
        (72, 150),
        "The observer pattern notifies dependents of state changes.",
        fontsize=12,
        fontname="helv",
    )
    doc.save(str(path))
    doc.close()
    return path


def _create_test_epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-002")
    book.set_title("Test EPUB")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Ch1", file_name="ch01.xhtml", lang="en")
    chapter.content = """<html><body>
        <h1>EPUB Chapter</h1>
        <p>Content about design patterns in software.</p>
    </body></html>"""
    book.add_item(chapter)
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def pipeline(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        books_dir=tmp_path / "data" / "books",
        chroma_db_dir=tmp_path / "data" / "chroma_db",
    )
    return IngestionPipeline(
        parsers=[PdfParser(), EpubParser()],
        chunker=SemanticChunker(settings),
        embedder=SentenceTransformerEmbedder(settings),
        vector_store=ChromaVectorStore(settings),
        settings=settings,
    )


def test_ingest_pdf(pipeline, tmp_path):
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    count = pipeline.ingest_file(pdf_path, "test")
    assert count > 0
    assert pipeline.vector_store.count("test") == count


def test_ingest_epub(pipeline, tmp_path):
    epub_path = _create_test_epub(tmp_path / "test.epub")
    count = pipeline.ingest_file(epub_path, "test")
    assert count > 0


def test_ingest_directory(pipeline, tmp_path):
    _create_test_pdf(tmp_path / "a.pdf")
    _create_test_epub(tmp_path / "b.epub")
    count = pipeline.ingest_directory(tmp_path, "test")
    assert count > 0


def test_force_reingest(pipeline, tmp_path):
    pdf_path = _create_test_pdf(tmp_path / "test.pdf")
    count1 = pipeline.ingest_file(pdf_path, "test")
    count2 = pipeline.ingest_file(pdf_path, "test", force=True)
    assert count2 == count1  # Same content, same chunks


def test_unsupported_file(pipeline, tmp_path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("hello")
    count = pipeline.ingest_file(txt_path, "test")
    assert count == 0
