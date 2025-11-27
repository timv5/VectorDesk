CREATE SCHEMA IF NOT EXISTS vector_desk;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE vector_desk.document_chunks (
     id          bigserial PRIMARY KEY,
     doc_id      text NOT NULL,
     chunk_index int  NOT NULL,
     content     text NOT NULL,
     embedding   vector(384) NOT NULL   -- dimension must match your embedding model
);
