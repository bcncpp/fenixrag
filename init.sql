-- Initialize the database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Optimize for vector operations
SET maintenance_work_mem = '512MB';

-- Create indexes for better performance
-- (These will be created by SQLAlchemy models, but can be pre-created here)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255),
    content TEXT,
    source VARCHAR(255),
    document_type VARCHAR(50),
    doc_metadata JSONB,
    embedding vector(768), 
    is_processed BOOLEAN DEFAULT FALSE,
    processing_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE document_metadata (
    id UUID PRIMARY KEY,
    document_id UUID,
    key VARCHAR,
    value VARCHAR,
    value_type VARCHAR,
    created_at TIMESTAMP WITHOUT TIME ZONE
);


-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_is_processed ON documents(is_processed);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
