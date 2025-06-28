-- Initialize the database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Optimize for vector operations
SET maintenance_work_mem = '512MB';

-- Create indexes for better performance
-- (These will be created by SQLAlchemy models, but can be pre-created here)
