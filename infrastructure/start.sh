#!/bin/sh
set -e

echo "Waiting for Postgres..."
until pg_isready -h pgvector-db -p 5432 -U localmind > /dev/null 2>&1; do
  sleep 1
done

echo "Postgres is up. Running indexing..."
python index_docs.py

echo "Starting QA API..."
uvicorn qa:app --host 0.0.0.0 --port 8000
