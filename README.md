# VectorDesk
Local, Private, Document-Based Question-Answering System

## Description
VectorDesk is an entirely local AI-powered assistant designed to answer questions using your internal company documents.
It combines semantic search, embeddings, and a local Large Language Model (LLM) to provide accurate answers without sending any data outside your machine.

This project is a fully private and fully offline alternative to cloud-based AI assistants.

## How does it work? 
User question → Embedding → Vector DB → Retrieve top chunks → Construct prompt → LLM generates answer → Return to user

## How can you run it?
in the root project simply run: 
```./run_stack.sh```

The system is split into two independent Python processors:

### **1. embeddings-processor**
- Reads `.docx` files from a defined folder /data
- Extracts paragraph text
- Splits text into overlapping chunks
- Generates embeddings via a local SentenceTransformers model
- Stores chunks + embeddings into PostgreSQL (`vector_desk.document_chunks`)

### **2. llm-processor**
- Embeds the user’s question using the same embedding model
- Retrieves the most similar document chunks using pgvector distance
- Builds a prompt containing retrieved context
- Sends the prompt to a local LLM managed by Ollama
- Returns the generated answer and lists the source chunks

## Requirements
- Python 3.10+
- Docker (PostgreSQL)
- PostgreSQL 16 + `pgvector` extension
- Ollama installed locally
- `.docx` files as input documents

## Settings
Located in /infrastructure folder.

### Database DSN
- DB_DSN = "dbname=localmind user=localmind password=localmind host=localhost port=5434"

### Embedding model
- EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

### LLM model (Ollama)
- OLLAMA_MODEL = "llama3.2"

### Retrieval depth
- top_k = 5

### Notes
- Embedding dimension must match the table definition (vector(384) for MiniLM)
- All processing remains local; no network calls except to the locally running Ollama service
- The ingestion processor can be rerun when documents change
- Retrieval uses pgvector <-> distance for similarity ranking

## Setup

### 1. Start Postgres with pgvector

From the `infrastructure/` directory:

```
docker-compose up -d
```

### 2. Prepare Python environment
- ```python3 -m venv .venv```
- ```source .venv/bin/activate```
- ```pip install -r requirements.txt```

### 3. Place .docx files in:
- embeddings-processor/data/

### 4. Run ingestion script
```python embeddings-processor/index_docs.py```

### 5. Set Ollama and verify
```ollama pull llama3.2```
```curl http://localhost:11434/api/tags```

### 6. Start QA processor
```python llm-processor/qa.py```

### 7. Start http server
```cd /llm-processor```
And then run either (api or cli mode):
```python qa.py --mode cli```
```python qa.py --mode api```


## CLI Example
You: What does the policy say about vacations?
Bot: ...
Sources:
- hr_policy.docx (chunk 3)

## API Example
Request:
```
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What is VectorDesk?", "top_k": 5}'
```

Response:
```
{
  "answer": "Lorem Ipsum is simply dummy text of the printing and typesetting industry, originating from sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" by Cicero, written in 45 BC.",
  "sources": [
    {
      "doc_id": "dummy.docx",
      "chunk_index": 2
    },
    {
      "doc_id": "dummy.docx",
      "chunk_index": 0
    },
    {
      "doc_id": "dummy.docx",
      "chunk_index": 3
    },
    {
      "doc_id": "dummy.docx",
      "chunk_index": 1
    }
  ]
}
```
