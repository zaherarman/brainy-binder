# Brainy Binder

Brainy Binder is a local-first, privacy-focused AI knowledge assistant that enables semantic search, question answering, summarization, and tagging over personal documents.

All data processing, embedding, retrieval, and language model inference runs locally. No documents or queries leave the machine.


## What This Project Does

Brainy Binder allows you to:

- Ingest personal documents (PDF, TXT, Markdown, DOCX)
- Chunk and embed document content
- Store embeddings in a local Chroma vector database
- Store document metadata in a local SQLite database
- Ask natural language questions over your documents using RAG
- Run a local LLM (Mistral via Ollama) for answering and summarization

The system is designed to be transparent, debuggable, and extensible.


## High-Level Architecture

1. Documents are discovered from the data directory
2. Each document is loaded and split into chunks
3. Chunks are embedded and stored in ChromaDB
4. Document-level metadata is stored in SQLite
5. Queries perform semantic search over vectors
6. Retrieved context is passed to a local LLM for generation


## Project Directory Structure
```
brainy-binder/
│
├── cli.py
├── config.py
│
├── data/
│
├── db/
│   ├── models.py
│   ├── session.py
│   └── brainy_binder.db
│
├── ingestion/
│   ├── pipeline.py
│   ├── loaders.py
│   └── chunking.py
│
├── vectorstore/
│   ├── chroma_store.py
│   └── embeddings.py
│
├── rag/
│   └── answer_engine.py
│
├── llm/
│   ├── client.py
│   └── prompts.py
│
├── agents/
│   └── semantic_tagging.py
│
└── README.md
```