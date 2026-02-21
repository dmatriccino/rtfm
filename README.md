# RTFM — Technical Book RAG System

Ingest technical books (PDF/EPUB), chunk them intelligently, generate embeddings, store them in a vector database, and retrieve relevant passages at query time. Built for feeding software engineering knowledge to AI coding agents via CLI and MCP server.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### 1. Ingest a book

```bash
# Single file
rtfm ingest data/books/design-patterns.pdf

# Entire directory
rtfm ingest data/books/

# Force re-ingest
rtfm ingest data/books/clean-code.epub --force

# Custom collection
rtfm ingest data/books/go-book.pdf --collection golang
```

### 2. Query the knowledge base

```bash
rtfm query "what is the strategy pattern"

# More results
rtfm query "dependency injection" --top-k 10

# Filter by book
rtfm query "error handling" --book "Clean Code"

# Code-only results
rtfm query "observer pattern implementation" --code
```

### 3. List and inspect

```bash
rtfm list         # Show ingested sources
rtfm stats        # Show collection statistics
```

## MCP Server Setup

RTFM includes an MCP server for use with Claude Code and other AI agents.

### Configure in Claude Code

The repo includes `.claude/mcp.json`. After installing the package (`pip install -e .`), Claude Code will automatically discover the `rtfm-mcp` server.

Alternatively, add manually:

```json
{
  "mcpServers": {
    "rtfm": {
      "command": "rtfm-mcp"
    }
  }
}
```

### Available MCP Tools

- **`query_knowledge`** — Search for relevant text passages
- **`search_code`** — Search specifically for code snippets
- **`list_books`** — List ingested sources

## Architecture

```
rtfm/
├── config/settings.py          # Pydantic-settings configuration
├── rtfm/
│   ├── models/schemas.py       # Data models (Section, Chunk, QueryResult, etc.)
│   ├── ingestion/
│   │   ├── base.py             # DocumentParser ABC
│   │   ├── pdf_parser.py       # PyMuPDF-based PDF parser
│   │   ├── epub_parser.py      # ebooklib + BeautifulSoup EPUB parser
│   │   ├── chunker.py          # Semantic chunking with prose/code strategies
│   │   └── pipeline.py         # Orchestrates parse → chunk → embed → store
│   ├── embeddings/
│   │   ├── base.py             # EmbeddingProvider ABC
│   │   └── local.py            # Sentence-transformers (all-MiniLM-L6-v2)
│   ├── storage/
│   │   ├── base.py             # VectorStore ABC
│   │   └── chroma.py           # ChromaDB with cosine similarity
│   └── retrieval/
│       └── retriever.py        # Query embedding + vector search
├── cli/main.py                 # Typer CLI
├── mcp_server/server.py        # FastMCP server
└── tests/                      # Comprehensive test suite
```

### Pipeline Flow

1. **Parse** — Extract structured sections (headings, prose, code) from PDF/EPUB
2. **Chunk** — Split into embedding-sized chunks with metadata preservation
3. **Embed** — Generate 384-dim vectors via all-MiniLM-L6-v2
4. **Store** — Upsert into ChromaDB with cosine distance
5. **Retrieve** — Embed query, search, rank by similarity

## Configuration

All settings can be overridden via environment variables with `RTFM_` prefix or a `.env` file:

| Setting | Default | Env Var |
|---------|---------|---------|
| `data_dir` | `data` | `RTFM_DATA_DIR` |
| `books_dir` | `data/books` | `RTFM_BOOKS_DIR` |
| `chroma_db_dir` | `data/chroma_db` | `RTFM_CHROMA_DB_DIR` |
| `embedding_model` | `all-MiniLM-L6-v2` | `RTFM_EMBEDDING_MODEL` |
| `embedding_dimensions` | `384` | `RTFM_EMBEDDING_DIMENSIONS` |
| `max_chunk_size` | `1500` | `RTFM_MAX_CHUNK_SIZE` |
| `max_code_chunk_size` | `3000` | `RTFM_MAX_CODE_CHUNK_SIZE` |
| `chunk_overlap` | `200` | `RTFM_CHUNK_OVERLAP` |
| `default_top_k` | `5` | `RTFM_DEFAULT_TOP_K` |
| `default_collection` | `engineering` | `RTFM_DEFAULT_COLLECTION` |
| `log_level` | `INFO` | `RTFM_LOG_LEVEL` |

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .

# Format
ruff format .
```
