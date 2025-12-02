#!/bin/bash
set -e

cd "$(dirname "$0")/infrastructure"

echo "Building and starting stack..."
docker compose up -d --build

echo "Pulling Ollama model llama3.2..."
docker exec ollama ollama pull llama3.2

echo "Stack is up. FastAPI should be on http://localhost:8000"
