"""Database package"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from ..config import settings

logger = logging.getLogger(__name__)


# Synchronous engine (for migrations and setup)
sync_engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=settings.connection_pool_size,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    echo=(settings.log_level == "DEBUG"),
)

# Asynchronous engine (for runtime operations)
async_engine = create_async_engine(
    settings.async_database_url,
    poolclass=QueuePool,
    pool_size=settings.connection_pool_size,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    echo=(settings.log_level == "DEBUG"),
)

# Session factories
SyncSessionLocal = sessionmaker(bind=sync_engine)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


class DatabaseManager:
    """Database management utilities"""

    @staticmethod
    def initialize_database():
        """Initialize database with pgvector extension and create tables"""
        try:
            with sync_engine.connect() as connection:
                # Enable pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                connection.commit()
                logger.info("pgvector extension enabled")

                # Import models to ensure they're registered
                from .models import Document, DocumentChunk, DocumentMetadata

                # Create all tables
                Base.metadata.create_all(bind=sync_engine)
                logger.info("Database tables created")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @staticmethod
    async def check_connection():
        """Check database connection"""
        try:
            async with async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    @staticmethod
    async def close_connections():
        """Close all database connections"""
        await async_engine.dispose()
        sync_engine.dispose()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get sync database session"""
    return SyncSessionLocal()
