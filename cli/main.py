"""RTFM CLI — ingest, query, and manage your technical knowledge base."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import get_settings
from rtfm.embeddings.local import SentenceTransformerEmbedder
from rtfm.ingestion.chunker import SemanticChunker
from rtfm.ingestion.epub_parser import EpubParser
from rtfm.ingestion.pdf_parser import PdfParser
from rtfm.ingestion.pipeline import IngestionPipeline
from rtfm.retrieval.retriever import Retriever
from rtfm.storage.chroma import ChromaVectorStore

app = typer.Typer(name="rtfm", help="Technical Book RAG System")
console = Console()


def _build_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        parsers=[PdfParser(), EpubParser()],
        chunker=SemanticChunker(settings),
        embedder=SentenceTransformerEmbedder(settings),
        vector_store=ChromaVectorStore(settings),
        settings=settings,
    )


def _build_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        embedding_provider=SentenceTransformerEmbedder(settings),
        vector_store=ChromaVectorStore(settings),
        settings=settings,
    )


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest existing files"),
):
    """Ingest a PDF/EPUB file or directory of files."""
    pipeline = _build_pipeline()

    if path.is_dir():
        count = pipeline.ingest_directory(path, collection, force)
    elif path.is_file():
        count = pipeline.ingest_file(path, collection, force)
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Ingested {count} chunks[/green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results"),
    book: str = typer.Option(None, "--book", "-b", help="Filter by book title"),
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name"),
    code: bool = typer.Option(False, "--code", help="Only return code results"),
):
    """Search the knowledge base."""
    retriever = _build_retriever()

    content_type = "code" if code else None
    response = retriever.query(
        query_text=question,
        top_k=top_k,
        book_title=book,
        content_type=content_type,
        collection_name=collection,
    )

    if not response.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, result in enumerate(response.results, 1):
        heading = result.metadata.get("heading", "")
        source = result.metadata.get("source_file", "")
        book_title = result.metadata.get("book_title", "")
        score = f"{result.score:.3f}"

        title = f"[{i}] {heading}" if heading else f"[{i}]"
        subtitle = f"{book_title} | {Path(source).name} | score: {score}"

        console.print(Panel(result.content, title=title, subtitle=subtitle, expand=False))


@app.command(name="list")
def list_books(
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name"),
):
    """List ingested source files."""
    settings = get_settings()
    store = ChromaVectorStore(settings)
    collection_name = collection or settings.default_collection

    sources = store.list_sources(collection_name)

    if not sources:
        console.print("[yellow]No sources found.[/yellow]")
        return

    table = Table(title="Ingested Sources")
    table.add_column("#", style="dim")
    table.add_column("Source File")

    for i, source in enumerate(sources, 1):
        table.add_row(str(i), source)

    console.print(table)


@app.command()
def stats(
    collection: str = typer.Option(None, "--collection", "-c", help="Collection name"),
):
    """Show collection statistics."""
    settings = get_settings()
    store = ChromaVectorStore(settings)
    collection_name = collection or settings.default_collection

    total_chunks = store.count(collection_name)
    sources = store.list_sources(collection_name)

    table = Table(title="Collection Stats")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Collection", collection_name)
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Total Sources", str(len(sources)))

    console.print(table)


if __name__ == "__main__":
    app()
