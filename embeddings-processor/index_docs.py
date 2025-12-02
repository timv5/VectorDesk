import os
from typing import List

import psycopg2
from psycopg2.extras import execute_values
from docx import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
DB_DSN = os.getenv(
    "DB_DSN",
    "dbname=localmind user=localmind password=localmind host=localhost port=5434",
)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")


def load_docx_text(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)

    if n == 0:
        return chunks

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == n:
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks



def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    for filename in os.listdir(DATA_DIR):
        if not filename.lower().endswith(".docx"):
            continue

        path = os.path.join(DATA_DIR, filename)
        doc_id = filename

        print(f"Processing {doc_id}...")

        text = load_docx_text(path)
        if not text.strip():
            print(f"  {doc_id} is empty or has no text, skipping.")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks created, generating embeddings...")

        embeddings = embedder.encode(chunks, convert_to_numpy=True)
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append((doc_id, i, chunk, emb.tolist()))

        insert_sql = """
                     INSERT INTO vector_desk.document_chunks (doc_id, chunk_index, content, embedding)
                     VALUES %s \
                     """

        template = "(%s, %s, %s, %s::vector)"

        rows_converted = []
        for doc_id, i, chunk, emb_list in rows:
            pg_array = "[" + ",".join(str(x) for x in emb_list) + "]"
            rows_converted.append((doc_id, i, chunk, pg_array))

        execute_values(cur, insert_sql, rows_converted, template=template)
        conn.commit()

    cur.close()
    conn.close()
    print("Done indexing .docx files.")


if __name__ == "__main__":
    main()
