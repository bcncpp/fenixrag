import json
from typing import Any
from uuid import UUID

from langchain_core.documents import Document as LCDocument
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..common.logger import LoggingMixin
from ..database import get_async_session
from ..database.models import Document, DocumentMetadata
from ..embeddings.gemini_embeddings import GeminiEmbeddings


class DocumentService(LoggingMixin):
    """Service for document operations with Gemini embeddings."""

    def __init__(self):
        """Initialize the document service."""
        self.embeddings = GeminiEmbeddings()

    async def add_document(
        self,
        content: str,
        title: str | None = None,
        source: str | None = None,
        document_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Add a single document."""
        try:
            # Generate embedding
            embedding = await self.embeddings.aembed_query(content)

            async with get_async_session() as session:
                # Create document
                document = Document(
                    title=title,
                    content=content,
                    source=source,
                    document_type=document_type,
                    metadata=metadata or {},
                    embedding=embedding,
                    is_processed=True,
                )

                session.add(document)
                await session.commit()
                await session.refresh(document)

                # Add metadata entries if provided
                if metadata:
                    await self._add_metadata_entries(session, document.id, metadata)
                    await session.commit()

                self.log.info(f"✅ Added document: {document.id}")
                return document.id

        except Exception as e:
            self.log.error(f"❌ Error adding document: {e}")
            raise

    async def add_documents_batch(self, documents_data: list[dict[str, Any]]) -> list[UUID]:
        """Add multiple documents in batch."""
        try:
            # Extract content for batch embedding
            contents = [doc.get("content", "") for doc in documents_data]

            # Generate embeddings in batch
            self.log.info(f"🔄 Generating embeddings for {len(contents)} documents...")
            embeddings = await self.embeddings.aembed_documents(contents)

            async with get_async_session() as session:
                document_ids = []

                for doc_data, embedding in zip(documents_data, embeddings):
                    # Create document
                    document = Document(
                        title=doc_data.get("title"),
                        content=doc_data.get("content", ""),
                        source=doc_data.get("source"),
                        document_type=doc_data.get("document_type"),
                        metadata=doc_data.get("metadata", {}),
                        embedding=embedding,
                        is_processed=True,
                    )

                    session.add(document)
                    document_ids.append(document.id)

                await session.commit()

                # Add metadata entries
                for doc_data, doc_id in zip(documents_data, document_ids):
                    if doc_data.get("metadata"):
                        await self._add_metadata_entries(session, doc_id, doc_data["metadata"])

                await session.commit()

                self.log.info(f"Added {len(document_ids)} documents")
                return document_ids

        except Exception as e:
            self.log.error(f"Error adding documents batch: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)

            async with get_async_session() as session:
                # Build base query
                query_stmt = select(
                    Document,
                    Document.embedding.cosine_distance(query_embedding).label("distance"),
                ).where(Document.is_processed == True)

                # Apply filters
                if filters:
                    filter_conditions = []

                    for key, value in filters.items():
                        if key in ["source", "document_type"]:
                            # Direct column filters
                            filter_conditions.append(getattr(Document, key) == value)
                        else:
                            # Metadata filters
                            filter_conditions.append(Document.metadata[key].astext == str(value))

                    if filter_conditions:
                        query_stmt = query_stmt.where(and_(*filter_conditions))

                # Apply similarity threshold
                if similarity_threshold:
                    distance_threshold = 1 - similarity_threshold
                    query_stmt = query_stmt.where(
                        Document.embedding.cosine_distance(query_embedding) <= distance_threshold
                    )

                # Order by similarity and limit
                query_stmt = query_stmt.order_by("distance").limit(limit)

                # Execute query
                result = await session.execute(query_stmt)
                documents_with_distance = result.fetchall()

                # Convert distance to similarity
                documents_with_similarity = [(doc, 1 - distance) for doc, distance in documents_with_distance]

                self.log.debug(f"🔍 Found {len(documents_with_similarity)} similar documents")
                return documents_with_similarity

        except Exception as e:
            self.log.error(f"Error in similarity search: {e}")
            raise

    async def get_documents_by_metadata(self, metadata_filters: dict[str, Any]) -> list[Document]:
        """Get documents by metadata filters."""
        try:
            async with get_async_session() as session:
                query_stmt = select(Document).where(Document.is_processed == True)

                # Apply metadata filters
                for key, value in metadata_filters.items():
                    query_stmt = query_stmt.where(Document.metadata[key].astext == str(value))

                result = await session.execute(query_stmt)
                documents = result.scalars().all()

                self.log.debug(f"Found {len(documents)} documents matching metadata filters")
                return documents

        except Exception as e:
            self.log.error(f"Error getting documents by metadata: {e}")
            raise

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document"""
        try:
            async with get_async_session() as session:
                # Get document
                stmt = select(Document).where(Document.id == document_id)
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()

                if document:
                    await session.delete(document)
                    await session.commit()
                    self.log.info(f"Deleted document: {document_id}")
                    return True
                else:
                    self.log.warning(f" Document not found: {document_id}")
                    return False

        except Exception as e:
            self.log.error(f"Error deleting document: {e}")
            raise

    async def _add_metadata_entries(self, session: AsyncSession, document_id: UUID, metadata: dict[str, Any]):
        """Add metadata entries for a document"""
        for key, value in metadata.items():
            # Determine value type
            if isinstance(value, bool):
                value_type = "boolean"
                value_str = str(value).lower()
            elif isinstance(value, (int, float)):
                value_type = "number"
                value_str = str(value)
            elif isinstance(value, (dict, list)):
                value_type = "json"
                value_str = json.dumps(value)
            else:
                value_type = "string"
                value_str = str(value)

            metadata_entry = DocumentMetadata(
                document_id=document_id,
                key=key,
                value=value_str,
                value_type=value_type,
            )
            session.add(metadata_entry)

    def to_langchain_documents(self, documents: list[Document]) -> list[LCDocument]:
        """Convert SQLAlchemy documents to LangChain documents"""
        lc_documents = []
        for doc in documents:
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata.update(
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "source": doc.source,
                    "document_type": doc.document_type,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                }
            )

            lc_doc = LCDocument(page_content=doc.content, metadata=metadata)
            lc_documents.append(lc_doc)

        return lc_documents
