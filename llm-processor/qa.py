import os
import json
from typing import List, Tuple, Any

import psycopg2
from sentence_transformers import SentenceTransformer
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === CONFIG ===
DB_DSN = os.getenv(
    "DB_DSN",
    "dbname=localmind user=localmind password=localmind host=localhost port=5434",
)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

embedder = SentenceTransformer(EMBED_MODEL_NAME)

app = FastAPI(title="VectorDesk QA API")

def embed_text(text: str) -> str:
    """
    Create embedding for text and return it as a Postgres array literal string,
    e.g. '[0.1,0.2,...]'.
    """
    vec = embedder.encode(text).tolist()
    pg_array = "[" + ",".join(str(x) for x in vec) + "]"
    return pg_array


def retrieve_context(question: str, top_k: int = 5) -> List[Tuple[Any, ...]]:
    """
    Retrieve the top_k most similar chunks from vector_desk.document_chunks
    based on the question embedding.
    Returns list of (doc_id, chunk_index, content).
    """
    pg_array = embed_text(question)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    query = """
            SELECT doc_id, chunk_index, content
            FROM vector_desk.document_chunks
            ORDER BY embedding <-> %s::vector
        LIMIT %s \
            """
    cur.execute(query, (pg_array, top_k))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def call_llm(prompt: str) -> str:
    """
    Call local Ollama model with the given prompt and return the full text response.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt},
        stream=True,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error: {resp.status_code} {resp.text}")

    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode("utf-8"))
        chunk = data.get("response", "")
        text += chunk
        if data.get("done"):
            break

    return text


def build_prompt(question: str, context_rows: List[Tuple[str, int, str]]) -> str:
    """
    Build the prompt to send to the LLM: instructions + retrieved chunks + question.
    context_rows: list of (doc_id, chunk_index, content).
    """
    context_str = ""
    for i, (doc_id, chunk_index, content) in enumerate(context_rows, start=1):
        context_str += f"[{i}] (from {doc_id}, chunk {chunk_index})\n{content}\n\n"

    prompt = f"""You are an internal company assistant.

                Use ONLY the information from the documents below to answer the user's question.
                If the answer is not present in the documents, say exactly:
                "I don't know based on the provided documents."
                
                Be concise and clear.
                
                === DOCUMENTS ===
                {context_str}
                === QUESTION ===
                {question}
                
                === ANSWER ===
            """
    return prompt


def answer(question: str, top_k: int = 5) -> Tuple[str, List[Tuple[Any, ...]]]:
    """
    High-level helper to get an answer + context rows.
    """
    context_rows = retrieve_context(question, top_k=top_k)
    if not context_rows:
        return "I couldn't find any relevant documents.", []

    prompt = build_prompt(question, context_rows)
    response_text = call_llm(prompt)
    return response_text.strip(), context_rows


# === HTTP API MODELS ===

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    doc_id: Any
    chunk_index: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# === HTTP ENDPOINT ===

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    POST /ask
    {
      "question": "your question",
      "top_k": 5
    }
    """
    try:
        answer_text, ctx_rows = answer(req.question, top_k=req.top_k)
        sources = [
            Source(doc_id=row[0], chunk_index=row[1])
            for row in ctx_rows
        ]
        return AskResponse(answer=answer_text, sources=sources)
    except Exception as e:
        # You can log here if you want
        raise HTTPException(status_code=500, detail=str(e))



def run_cli():
    print("Local QA bot. Type your question, or 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if not q or q.lower() in {"exit", "quit"}:
            break

        print("Thinking...")
        try:
            ans, ctx = answer(q)
        except Exception as e:
            print("Error:", e)
            continue

        print("\nBot:", ans)
        print("\nSources (documents used):")
        for doc_id, chunk_index, _ in ctx:
            print(f" - {doc_id} (chunk {chunk_index})")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "api"], default="cli")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        uvicorn.run("qa:app", host="0.0.0.0", port=8000, reload=False)
