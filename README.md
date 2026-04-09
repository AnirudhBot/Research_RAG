# Research Paper Assistant

RAG-based research assistant that lets you upload PDFs and ask questions about them. Extracts text, tables, and images from papers and uses LLM-powered chat with semantic search for context-aware responses.

## Features

- **Text extraction** — PyMuPDF for fast, accurate text parsing across all pages
- **Table extraction** — Hybrid PyMuPDF + pdfplumber pipeline with multi-pass strategies (lines_strict, lines, text) and noise filtering; tables stored as markdown for native LLM reasoning
- **Image awareness** — Extracts embedded figures/charts via PyMuPDF, generates searchable text descriptions using GPT-4o vision, optional OpenCV preprocessing for low-quality images
- **Vector search** — Qdrant vector database with text-embedding-3-small for semantic retrieval of relevant document fragments
- **Conversational RAG** — History-aware retrieval chain that reformulates follow-up questions and synthesises answers across text, tables, and figures

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini (chat), GPT-4o (vision) |
| Embeddings | text-embedding-3-small |
| Vector DB | Qdrant (cloud or in-memory) |
| PDF parsing | PyMuPDF, pdfplumber |
| Image processing | OpenCV, Pillow |
| Orchestration | LangChain 0.3.x |

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY="sk-proj-..."              # Required
QDRANT_HOST="https://...qdrant.io"        # Optional — omit for in-memory mode
QDRANT_API_KEY="..."                      # Required if QDRANT_HOST is set
AI_MODEL="gpt-4o-mini"                   # Chat model (default: gpt-4o-mini)
IMAGE_DESCRIPTION_MODEL="gpt-4o"          # Vision model for figures
```

### 3. Run

```bash
streamlit run src/main.py
```

Opens at `http://localhost:8501`.

### Docker

```bash
docker build -t research-rag .
docker run -p 8501:8501 --env-file .env research-rag
```

## Usage

1. Upload one or more research paper PDFs via the sidebar
2. Click **Process Documents** — extracts text, tables, images and indexes them
3. Ask questions in the chat
