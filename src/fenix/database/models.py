from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import settings

# Create base class for models
Base = declarative_base()

class Document(Base):
    """Main document table"""
    
    __tablename__ = "documents"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Document content
    title = Column(String(500))
    content = Column(Text, nullable=False)
    source = Column(String(500))
    document_type = Column(String(100))
    doc_metadata = Column(JSON, default=dict)
    
    # Vector embedding
    embedding = Column(Vector(settings.vector_dimension))
    
    # Status and processing info
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    metadata_entries = relationship("DocumentMetadata", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes for vector operations
    __table_args__ = (
        Index(
            'ix_documents_embedding_cosine',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index(
            'ix_documents_embedding_l2',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
        Index('ix_documents_source', 'source'),
        Index('ix_documents_document_type', 'document_type'),
        Index('ix_documents_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', source='{self.source}')>"


class DocumentChunk(Base):
    """Document chunks for large documents"""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Vector embedding for the chunk
    embedding = Column(Vector(settings.vector_dimension))
    
    # Chunk metadata
    start_char = Column(Integer)
    end_char = Column(Integer)
    chunk_size = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index(
            'ix_chunks_embedding_cosine',
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index('ix_chunks_document_id', 'document_id'),
        Index('ix_chunks_chunk_index', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class DocumentMetadata(Base):
    """Flexible metadata for documents"""
    
    __tablename__ = "document_metadata"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    
    # Metadata key-value pairs
    key = Column(String(255), nullable=False)
    value = Column(Text)
    value_type = Column(String(50), default='string')  # string, number, boolean, json
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="metadata_entries")
    
    # Indexes
    __table_args__ = (
        Index('ix_metadata_document_id', 'document_id'),
        Index('ix_metadata_key', 'key'),
        Index('ix_metadata_key_value', 'key', 'value'),
    )
    
    def __repr__(self):
        return f"<DocumentMetadata(key='{self.key}', value='{self.value}')>"

