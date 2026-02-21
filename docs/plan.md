# RTFM — Technical Book RAG System

## Context

Build a Python RAG system that ingests technical books (PDF/EPUB), chunks them intelligently, generates embeddings, stores them in a vector database, and retrieves relevant passages at query time. The primary use case is feeding software engineering knowledge to AI coding agents via CLI and MCP server. The repo is currently empty (just a placeholder README).

## Environment

- **Existing venv**: `/home/dmatriccino/workspace/rtfm/venv/` (Python 3.13.7)
- All `pip install` commands use `./venv/bin/pip`
- Target Python version: **3.13** (update ruff target accordingly)

## Structural Change from Spec

- **`src/` → `rtfm/`**: Standard Python packaging convention so imports are `from rtfm.ingestion import ...`
- **`mcp/` → `mcp_server/`**: A local `mcp/` directory shadows the `mcp` PyPI package. Must rename.

## Final Project Structure

```
rtfm/                           # repo root
├── pyproject.toml
├── .gitignore
├── README.md
├── config/
│   └── settings.py             # pydantic-settings config
├── rtfm/                       # main package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models: Section, Chunk, BookMetadata, QueryResult, QueryResponse
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py             # ABC: DocumentParser
│   │   ├── pdf_parser.py       # PyMuPDF implementation
│   │   ├── epub_parser.py      # ebooklib + BeautifulSoup implementation
│   │   ├── chunker.py          # SemanticChunker
│   │   └── pipeline.py         # IngestionPipeline orchestrator
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py             # ABC: EmbeddingProvider
│   │   └── local.py            # SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py             # ABC: VectorStore
│   │   └── chroma.py           # ChromaVectorStore (PersistentClient, cosine distance)
│   └── retrieval/
│       ├── __init__.py
│       └── retriever.py        # Retriever: embed query → search → return ranked results
├── cli/
│   ├── __init__.py
│   └── main.py                 # Typer CLI: ingest, query, list, stats
├── mcp_server/
│   ├── __init__.py
│   └── server.py               # FastMCP server: query_knowledge, list_books, search_code
├── data/
│   ├── books/                  # .gitignored — drop PDFs/EPUBs here
│   └── chroma_db/              # .gitignored — ChromaDB persistence
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_schemas.py
    ├── test_chunker.py
    ├── test_pdf_parser.py
    ├── test_epub_parser.py
    ├── test_embeddings.py
    ├── test_storage.py
    ├── test_retrieval.py
    ├── test_pipeline.py
    └── fixtures/
```

---

## Phase 1: Foundation — Packaging, Models, Config

### `pyproject.toml`
- Build system: setuptools
- Runtime deps: `pymupdf`, `ebooklib`, `beautifulsoup4`, `sentence-transformers`, `chromadb`, `pydantic>=2.0`, `pydantic-settings>=2.0`, `typer>=0.12`, `rich>=13.0`, `mcp[cli]>=1.0`, `structlog`
- Dev deps: `pytest`, `pytest-asyncio`, `ruff`
- Entry points: `rtfm = "cli.main:app"`, `rtfm-mcp = "mcp_server.server:main"`
- Ruff: line-length=100, target py313
- Pytest: testpaths=["tests"]

### `rtfm/models/schemas.py`
- `ContentType(str, Enum)`: PROSE, CODE, HEADING, TABLE, LIST
- `Section(BaseModel)`: Parser output — content, content_type, heading, heading_level, page_number, source_file, font_name, font_size
- `Chunk(BaseModel)`: Chunker output — id (sha256 hash of source_file+content, truncated to 16 chars), content, content_type, metadata dict
- `BookMetadata(BaseModel)`: title, file_path, file_type, total_pages, total_chunks, ingested_at
- `QueryResult(BaseModel)`: content, score, metadata dict
- `QueryResponse(BaseModel)`: query, results list, total_results

### `config/settings.py`
- `Settings(BaseSettings)` with `env_prefix="RTFM_"`, `.env` file support
- Paths: data_dir, books_dir, chroma_db_dir
- Embedding: model name, dimensions (384)
- Chunking: max_chunk_size (1500 chars), max_code_chunk_size (3000 chars), chunk_overlap (200 chars)
- Retrieval: default_top_k (5), default_collection ("engineering")
- Logging: log_level

### `.gitignore`
- Standard Python ignores + `venv/`, `data/chroma_db/`, `data/books/*.pdf`, `data/books/*.epub`, `.env`

---

## Phase 2: Abstract Interfaces

### `rtfm/ingestion/base.py` — `DocumentParser(ABC)`
- `can_parse(file_path: Path) -> bool`
- `parse(file_path: Path) -> tuple[BookMetadata, list[Section]]`

### `rtfm/embeddings/base.py` — `EmbeddingProvider(ABC)`
- `embed_texts(texts: list[str]) -> list[list[float]]` — batch embedding
- `embed_query(query: str) -> list[float]` — single query (separate for future query-specific prefixes)
- `dimensions -> int` — property

### `rtfm/storage/base.py` — `VectorStore(ABC)`
- `upsert_chunks(chunks, embeddings, collection_name) -> int`
- `query(query_embedding, top_k, where, collection_name) -> list[QueryResult]`
- `delete_by_source(source_file, collection_name) -> int`
- `list_sources(collection_name) -> list[str]`
- `count(collection_name) -> int`

---

## Phase 3: Parsers

### `rtfm/ingestion/pdf_parser.py` — `PdfParser(DocumentParser)`
- Uses `pymupdf` with `page.get_text("dict")` for per-span font info
- Code detection: monospace fonts (Courier, Mono, Consolas, Menlo)
- Heading detection: font size significantly larger than median
- TOC extraction via `doc.get_toc()` for heading assignment
- Post-processing: merge adjacent same-type spans, assign nearest heading to each section

### `rtfm/ingestion/epub_parser.py` — `EpubParser(DocumentParser)`
- Uses `ebooklib` to iterate `ITEM_DOCUMENT` items
- `BeautifulSoup` to parse HTML content
- `<pre>`/`<code>` (block-level) → CODE, `<h1>`-`<h6>` → HEADING, `<p>` → PROSE
- Tracks current heading state while walking DOM
- Metadata from `book.get_metadata("DC", "title")` and `("DC", "creator")`

---

## Phase 4: Chunker

### `rtfm/ingestion/chunker.py` — `SemanticChunker`
- Input: `list[Section]` + `BookMetadata` → Output: `list[Chunk]`
- Headings become metadata on subsequent chunks, not standalone chunks
- **Prose strategy**: Split by paragraph boundaries (double newline), then by sentence if still too large. Apply overlap (200 chars) between consecutive chunks.
- **Code strategy**: Keep code blocks intact if possible. If > max_code_chunk_size, split at blank lines. No overlap for code.
- Chunk ID: `sha256(f"{source_file}::{content}")[:16]` — deterministic for idempotent upsert
- Metadata dict per chunk: book_title, source_file, page_number, heading, content_type, char_count

---

## Phase 5: Concrete Implementations

### `rtfm/embeddings/local.py` — `SentenceTransformerEmbedder(EmbeddingProvider)`
- Loads `all-MiniLM-L6-v2` once in `__init__`
- `embed_texts`: `model.encode(texts, show_progress_bar=False).tolist()`
- `embed_query`: delegates to `embed_texts([query])[0]`
- `dimensions`: 384

### `rtfm/storage/chroma.py` — `ChromaVectorStore(VectorStore)`
- `PersistentClient(path=...)` with injected path
- `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`
- `upsert` in batches of 5000 (ChromaDB limit)
- Query returns cosine distance; convert to similarity: `score = 1.0 - distance`
- `delete_by_source`: get IDs by metadata filter, then delete
- `list_sources`: get all metadatas, extract unique source_file values

---

## Phase 6: Retriever & Pipeline

### `rtfm/retrieval/retriever.py` — `Retriever`
- Constructor injection: `EmbeddingProvider` + `VectorStore`
- `query(query_text, top_k, book_title, content_type, collection_name) -> QueryResponse`
- Builds ChromaDB `$and` filter from optional book_title and content_type params

### `rtfm/ingestion/pipeline.py` — `IngestionPipeline`
- Constructor injection: parsers list, chunker, embedder, vector store
- `ingest_file(path, collection, force)`: parse → chunk → embed (batch 256) → upsert. With `force`, deletes existing chunks first.
- `ingest_directory(path, collection, force)`: iterates files, auto-selects parser via `can_parse`
- Structured logging via `structlog` throughout
- Creates `data/` directories automatically with `mkdir(parents=True, exist_ok=True)`

---

## Phase 7: CLI

### `cli/main.py` — Typer app
- Factory functions `_build_pipeline()` and `_build_retriever()` wire up all dependencies
- **`rtfm ingest <path>`**: Ingest single file or directory. Options: `--collection`, `--force`
- **`rtfm query "<question>"`**: Search knowledge base. Options: `--top-k`, `--book`, `--collection`, `--code` (code-only filter). Output via Rich Panels.
- **`rtfm list`**: Table of ingested source files
- **`rtfm stats`**: Collection name, total chunks, total sources

---

## Phase 8: MCP Server

### `mcp_server/server.py` — FastMCP server
- `FastMCP("rtfm")` with lazy initialization (model loads on first tool call, not at startup)
- **`query_knowledge(query, namespace, top_k, book_title)`**: General search, returns formatted text passages
- **`search_code(query, namespace, top_k)`**: Filters for content_type="code", returns code snippets in fenced blocks
- **`list_books(namespace)`**: Lists ingested sources
- Entry point: `mcp.run(transport="stdio")`

Claude Code config (`.claude/mcp.json`):
```json
{
  "mcpServers": {
    "rtfm": {
      "command": "rtfm-mcp"
    }
  }
}
```

---

## Phase 9: Tests & Polish

### Tests
- `test_schemas.py`: Model construction, validation, chunk ID determinism
- `test_chunker.py`: Prose splitting, code preservation, heading propagation, overlap, large code splitting
- `test_pdf_parser.py`: Span classification, heading detection, section merging (programmatically create test PDF with pymupdf)
- `test_epub_parser.py`: Heading/code/prose extraction from minimal test EPUB
- `test_embeddings.py`: Correct dimension (384), batch consistency, deterministic output
- `test_storage.py`: Upsert, query, delete_by_source, idempotency (tmp_path ChromaDB)
- `test_retrieval.py`: Filter building, mock-based retriever integration
- `test_pipeline.py`: End-to-end with real small fixtures
- `conftest.py`: Shared fixtures — tmp_path settings, sample data factories

### README.md
- Installation instructions, first-time ingestion walkthrough, CLI usage examples, MCP setup for Claude Code, architecture overview, configuration reference
