"""RTFM MCP server — expose knowledge base to AI coding agents."""

from mcp.server.fastmcp import FastMCP

from config.settings import get_settings
from rtfm.embeddings.local import SentenceTransformerEmbedder
from rtfm.retrieval.retriever import Retriever
from rtfm.storage.chroma import ChromaVectorStore

mcp = FastMCP("rtfm")

# Lazy initialization — model loads on first tool call
_retriever: Retriever | None = None
_store: ChromaVectorStore | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        settings = get_settings()
        embedder = SentenceTransformerEmbedder(settings)
        store = _get_store()
        _retriever = Retriever(embedder, store, settings)
    return _retriever


def _get_store() -> ChromaVectorStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = ChromaVectorStore(settings)
    return _store


@mcp.tool()
def query_knowledge(
    query: str,
    namespace: str = "engineering",
    top_k: int = 5,
    book_title: str | None = None,
) -> str:
    """Search the technical knowledge base for relevant passages.

    Args:
        query: The search query
        namespace: Collection namespace to search
        top_k: Number of results to return
        book_title: Optional filter by book title
    """
    retriever = _get_retriever()
    response = retriever.query(
        query_text=query,
        top_k=top_k,
        book_title=book_title,
        collection_name=namespace,
    )

    if not response.results:
        return "No results found."

    parts = []
    for i, result in enumerate(response.results, 1):
        heading = result.metadata.get("heading", "")
        source = result.metadata.get("source_file", "")
        book = result.metadata.get("book_title", "")
        header = f"[{i}] {heading}" if heading else f"[{i}]"
        body = f"{header}\nSource: {book} ({source})\nScore: {result.score:.3f}\n\n{result.content}"
        parts.append(body)

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_code(
    query: str,
    namespace: str = "engineering",
    top_k: int = 5,
) -> str:
    """Search for code snippets in the technical knowledge base.

    Args:
        query: The search query for code
        namespace: Collection namespace to search
        top_k: Number of results to return
    """
    retriever = _get_retriever()
    response = retriever.query(
        query_text=query,
        top_k=top_k,
        content_type="code",
        collection_name=namespace,
    )

    if not response.results:
        return "No code snippets found."

    parts = []
    for i, result in enumerate(response.results, 1):
        heading = result.metadata.get("heading", "")
        book = result.metadata.get("book_title", "")
        header = f"[{i}] {heading}" if heading else f"[{i}]"
        body = f"{header}\nSource: {book}\nScore: {result.score:.3f}\n\n```\n{result.content}\n```"
        parts.append(body)

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_books(namespace: str = "engineering") -> str:
    """List all ingested books/sources in the knowledge base.

    Args:
        namespace: Collection namespace to list
    """
    store = _get_store()
    sources = store.list_sources(namespace)

    if not sources:
        return "No books ingested yet."

    lines = [f"{i}. {source}" for i, source in enumerate(sources, 1)]
    return "\n".join(lines)


def main():
    """Entry point for rtfm-mcp command."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
