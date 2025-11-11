-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document_vectors table
CREATE TABLE IF NOT EXISTS document_vectors (
    id INTEGER PRIMARY KEY,
    embedding vector(384)
);

-- Drop and recreate document_chunk table
DROP TABLE IF EXISTS document_chunk;

CREATE TABLE document_chunk (
    id SERIAL PRIMARY KEY,
    doc_name VARCHAR(512) NOT NULL,
    page INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    text TEXT NOT NULL
);
