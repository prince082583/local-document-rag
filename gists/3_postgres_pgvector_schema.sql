-- PostgreSQL 18 + pgvector Enterprise Schema & HNSW Index

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create documents table with 768-dim vector column
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Build HNSW index for sub-millisecond approximate cosine similarity
-- m = 16 (bidirectional links per element), ef_construction = 64 (build exploration factor)
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
