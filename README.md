# VectorDesk
Local, Private, Document-Based Question-Answering System

## Description
VectorDesk is an entirely local AI-powered assistant designed to answer questions using your internal company documents.
It combines semantic search, embeddings, and a local Large Language Model (LLM) to provide accurate answers without sending any data outside your machine.

This project is a fully private and fully offline alternative to cloud-based AI assistants.

## How does it work? 
User question → Embedding → Vector DB → Retrieve top chunks → Construct prompt → LLM generates answer → Return to user


## Internal notes
- sudo apt update
- sudo apt python3 -m venv .venv
- source .venv/bin/activate
- pip install sentence-transformers python-docx psycopg2-binary // local embedding model, read word docs, connect to postgres

- ollama serve
- ollama pull llama3.2
- cd ~/Documents/personal/projects/VectorDesk
- source .venv/bin/activate
- pip install requests
