-- PostgreSQL Extensions Installation Script
-- This script runs on database initialization

-- Enable common extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_buffercache";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- For RAG applications - vector embeddings
CREATE EXTENSION IF NOT EXISTS "vector";

-- Additional extensions for prototyping
CREATE EXTENSION IF NOT EXISTS "hstore";
CREATE EXTENSION IF NOT EXISTS "tablefunc";

-- Create a sample table for testing
CREATE TABLE IF NOT EXISTS sample_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    vector_extension_available BOOLEAN DEFAULT TRUE,
    embedding VECTOR(3)  -- Match sample data dimensions
);

-- Create a dedicated table for vector embeddings
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES sample_data(id),
    content TEXT,
    embedding VECTOR(5),  -- Match sample data dimensions
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx
ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Insert sample data
INSERT INTO sample_data (data, embedding) VALUES
('{"type": "test", "content": "Sample data for RAG prototyping"}', '[0.1, 0.2, 0.3]'::VECTOR),
('{"type": "config", "content": "Database initialized with extensions"}', '[0.4, 0.5, 0.6]'::VECTOR);

-- Insert sample vector embeddings
INSERT INTO document_embeddings (content, embedding, metadata) VALUES
('This is a sample document for vector search', '[0.1, 0.2, 0.3, 0.4, 0.5]'::VECTOR, '{"source": "sample", "type": "document"}'),
('Another document with different content', '[0.6, 0.7, 0.8, 0.9, 0.1]'::VECTOR, '{"source": "sample", "type": "article"}');

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
GRANT ALL ON TABLE sample_data TO rag_user;
GRANT ALL ON TABLE document_embeddings TO rag_user;
