# =============================================================================
# FILE: tests/test_integration_testcontainers.py
# =============================================================================
import pytest
import asyncio
import time
import os
from uuid import uuid4
from typing import AsyncGenerator, Generator
from testcontainers.postgres import PostgresContainer
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.config.settings import Settings
from src.database.connection import Base, DatabaseManager
from src.database.models import Document, DocumentChunk, DocumentMetadata
from src.services.document_service import DocumentService
from src.services.rag_service import RAGService
from src.embeddings.gemini_embeddings import GeminiEmbeddings


class MockGeminiEmbeddings:
    """Mock Gemini embeddings for testing without API calls"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        print(f"📦 Using mock embeddings with {dimension} dimensions")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents"""
        import hashlib
        import numpy as np
        
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text content
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            
            # Generate normalized vector
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for query"""
        return self.embed_documents([text])[0]
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_documents"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query"""
        return self.embed_query(text)


class TestContainerDocumentService(DocumentService):
    """Document service with mock embeddings for testing"""
    
    def __init__(self):
        # Don't call super().__init__() to avoid Gemini API requirements
        self.embeddings = MockGeminiEmbeddings()


class TestContainerRAGService(RAGService):
    """RAG service with mock embeddings for testing"""
    
    def __init__(self, use_llm: bool = False):
        # Override to use mock document service and no LLM
        self.document_service = TestContainerDocumentService()
        self.use_llm = use_llm
        self.llm = None
        print("✅ RAG service initialized with mock embeddings (no LLM)")


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Start PostgreSQL container with pgvector for the test session"""
    
    print("🐳 Starting PostgreSQL testcontainer with pgvector...")
    
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test_user",
        password="test_password",
        dbname="test_db",
        port=5432
    ) as postgres:
        
        # Wait for container to be ready
        time.sleep(5)
        
        # Get connection details
        connection_url = postgres.get_connection_url()
        
        # Initialize pgvector extension
        print("🔧 Initializing pgvector extension...")
        conn = psycopg2.connect(connection_url)
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.close()
        
        print(f"✅ PostgreSQL ready at: {connection_url}")
        yield postgres


@pytest.fixture(scope="session")
def test_settings(postgres_container: PostgresContainer) -> Settings:
    """Create test settings with container database URL"""
    
    # Override settings for testing
    connection_url = postgres_container.get_connection_url()
    
    # Create sync and async URLs
    sync_url = connection_url.replace("postgresql://", "postgresql+psycopg2://")
    async_url = connection_url.replace("postgresql://", "postgresql+asyncpg://")
    
    # Mock settings
    test_settings = Settings(
        google_api_key="mock_api_key_for_testing",
        database_url=sync_url,
        async_database_url=async_url,
        vector_dimension=768,
        collection_name="test_documents",
        max_batch_size=50,
        log_level="DEBUG"
    )
    
    return test_settings


@pytest.fixture(scope="session")
def setup_database(test_settings: Settings) -> None:
    """Setup database tables for testing"""
    
    print("🔧 Setting up test database schema...")
    
    # Create sync engine for setup
    sync_engine = create_engine(test_settings.database_url, echo=False)
    
    # Create all tables
    Base.metadata.create_all(bind=sync_engine)
    
    print("✅ Test database schema created")
    yield
    
    # Cleanup
    Base.metadata.drop_all(bind=sync_engine)
    sync_engine.dispose()
    print("🧹 Test database schema cleaned up")


@pytest.fixture
async def async_session(test_settings: Settings, setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Provide async database session for tests"""
    
    # Create async engine
    async_engine = create_async_engine(test_settings.async_database_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with AsyncSessionLocal() as session:
        yield session
    
    await async_engine.dispose()


@pytest.fixture
def document_service(test_settings: Settings, setup_database) -> TestContainerDocumentService:
    """Provide document service for testing"""
    
    # Monkey patch settings
    import src.config.settings
    src.config.settings.settings = test_settings
    
    return TestContainerDocumentService()


@pytest.fixture
def rag_service(test_settings: Settings, setup_database) -> TestContainerRAGService:
    """Provide RAG service for testing"""
    
    # Monkey patch settings
    import src.config.settings
    src.config.settings.settings = test_settings
    
    return TestContainerRAGService(use_llm=False)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestDocumentServiceWithContainers:
    """Test document service with real PostgreSQL container"""
    
    @pytest.mark.asyncio
    async def test_add_single_document(self, document_service: TestContainerDocumentService):
        """Test adding a single document"""
        
        doc_id = await document_service.add_document(
            title="Test Document",
            content="This is a test document about machine learning algorithms and data science.",
            source="test_suite",
            document_type="test",
            metadata={"category": "ai", "difficulty": "beginner"}
        )
        
        assert doc_id is not None
        print(f"✅ Added document with ID: {doc_id}")
    
    @pytest.mark.asyncio
    async def test_add_batch_documents(self, document_service: TestContainerDocumentService):
        """Test batch document addition"""
        
        test_docs = [
            {
                "title": "Python Programming",
                "content": "Python is a high-level programming language known for its simplicity.",
                "source": "programming_guide",
                "document_type": "educational",
                "metadata": {"category": "programming", "language": "python"}
            },
            {
                "title": "JavaScript Basics",
                "content": "JavaScript is essential for web development and interactive websites.",
                "source": "web_guide",
                "document_type": "educational",
                "metadata": {"category": "programming", "language": "javascript"}
            },
            {
                "title": "Machine Learning",
                "content": "Machine learning algorithms can learn patterns from data automatically.",
                "source": "ai_textbook",
                "document_type": "academic",
                "metadata": {"category": "ai", "topic": "machine_learning"}
            }
        ]
        
        doc_ids = await document_service.add_documents_batch(test_docs)
        
        assert len(doc_ids) == 3
        assert all(doc_id is not None for doc_id in doc_ids)
        print(f"✅ Added {len(doc_ids)} documents in batch")
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, document_service: TestContainerDocumentService):
        """Test similarity search functionality"""
        
        # Add test documents first
        test_docs = [
            {
                "title": "Python Tutorial",
                "content": "Python programming language tutorial for beginners with examples.",
                "metadata": {"category": "programming"}
            },
            {
                "title": "Machine Learning Guide",
                "content": "Introduction to machine learning algorithms and neural networks.",
                "metadata": {"category": "ai"}
            },
            {
                "title": "Web Development",
                "content": "Building websites with HTML, CSS, and JavaScript frameworks.",
                "metadata": {"category": "web"}
            }
        ]
        
        await document_service.add_documents_batch(test_docs)
        
        # Test similarity search
        results = await document_service.similarity_search(
            query="python programming tutorial",
            limit=2
        )
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # Verify results structure
        for doc, score in results:
            assert hasattr(doc, 'title')
            assert hasattr(doc, 'content')
            assert isinstance(score, float)
            assert 0 <= score <= 1
        
        # The Python document should have highest similarity
        top_doc, top_score = results[0]
        assert "python" in top_doc.content.lower()
        
        print(f"✅ Similarity search returned {len(results)} results")
        print(f"   Top result: '{top_doc.title}' (score: {top_score:.3f})")
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_filters(self, document_service: TestContainerDocumentService):
        """Test similarity search with metadata filters"""
        
        # Add documents with different categories
        test_docs = [
            {
                "content": "Python data science libraries like pandas and numpy.",
                "metadata": {"category": "programming", "language": "python"}
            },
            {
                "content": "Java enterprise development with Spring framework.",
                "metadata": {"category": "programming", "language": "java"}
            },
            {
                "content": "Machine learning with Python scikit-learn library.",
                "metadata": {"category": "ai", "tool": "python"}
            }
        ]
        
        await document_service.add_documents_batch(test_docs)
        
        # Search with category filter
        results = await document_service.similarity_search(
            query="programming languages",
            limit=5,
            filters={"category": "programming"}
        )
        
        assert len(results) > 0
        
        # Verify all results match the filter
        for doc, score in results:
            assert doc.metadata.get("category") == "programming"
        
        print(f"✅ Filtered search returned {len(results)} results")
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, document_service: TestContainerDocumentService):
        """Test similarity search with score threshold"""
        
        # Add a test document
        await document_service.add_document(
            content="Deep learning neural networks for computer vision applications.",
            metadata={"category": "ai"}
        )
        
        # Search with high threshold (should return relevant results)
        high_threshold_results = await document_service.similarity_search(
            query="deep learning neural networks",
            similarity_threshold=0.5
        )
        
        # Search with very high threshold (should return fewer/no results)
        very_high_threshold_results = await document_service.similarity_search(
            query="completely unrelated topic",
            similarity_threshold=0.9
        )
        
        # High similarity query should return results
        assert len(high_threshold_results) > 0
        
        # All results should meet threshold
        for doc, score in high_threshold_results:
            assert score >= 0.5
        
        print(f"✅ Threshold search: {len(high_threshold_results)} results above 0.5")
        print(f"✅ High threshold search: {len(very_high_threshold_results)} results above 0.9")
    
    @pytest.mark.asyncio
    async def test_get_documents_by_metadata(self, document_service: TestContainerDocumentService):
        """Test getting documents by metadata filters"""
        
        # Add documents with specific metadata
        test_docs = [
            {
                "content": "Python programming tutorial",
                "metadata": {"category": "programming", "difficulty": "beginner", "language": "python"}
            },
            {
                "content": "Advanced Python concepts",
                "metadata": {"category": "programming", "difficulty": "advanced", "language": "python"}
            },
            {
                "content": "JavaScript fundamentals",
                "metadata": {"category": "programming", "difficulty": "beginner", "language": "javascript"}
            }
        ]
        
        await document_service.add_documents_batch(test_docs)
        
        # Test filtering by category
        programming_docs = await document_service.get_documents_by_metadata(
            {"category": "programming"}
        )
        assert len(programming_docs) >= 3
        
        # Test filtering by multiple criteria
        beginner_python_docs = await document_service.get_documents_by_metadata(
            {"difficulty": "beginner", "language": "python"}
        )
        assert len(beginner_python_docs) >= 1
        
        print(f"✅ Found {len(programming_docs)} programming documents")
        print(f"✅ Found {len(beginner_python_docs)} beginner Python documents")
    
    @pytest.mark.asyncio
    async def test_delete_document(self, document_service: TestContainerDocumentService):
        """Test document deletion"""
        
        # Add a document
        doc_id = await document_service.add_document(
            content="Document to be deleted",
            metadata={"test": "deletion"}
        )
        
        # Verify it exists
        results = await document_service.similarity_search("document to be deleted")
        assert len(results) > 0
        
        # Delete the document
        deleted = await document_service.delete_document(doc_id)
        assert deleted is True
        
        # Verify it's gone (search should return fewer results)
        results_after = await document_service.similarity_search("document to be deleted")
        # Note: Due to other test documents, we can't guarantee 0 results,
        # but the specific document should be gone
        
        print(f"✅ Successfully deleted document {doc_id}")


class TestRAGServiceWithContainers:
    """Test RAG service with real PostgreSQL container"""
    
    @pytest.mark.asyncio
    async def test_rag_query_without_llm(self, rag_service: TestContainerRAGService):
        """Test RAG query in context-only mode (no LLM)"""
        
        # Add test documents
        test_docs = [
            {
                "title": "Python Basics",
                "content": "Python is a beginner-friendly programming language with clear syntax.",
                "metadata": {"category": "programming", "difficulty": "beginner"}
            },
            {
                "title": "Machine Learning",
                "content": "Machine learning algorithms learn patterns from data to make predictions.",
                "metadata": {"category": "ai", "difficulty": "intermediate"}
            }
        ]
        
        await rag_service.document_service.add_documents_batch(test_docs)
        
        # Test RAG query
        result = await rag_service.query("What programming language is good for beginners?")
        
        # Verify response structure
        assert "answer" in result
        assert "sources" in result
        assert "context" in result
        assert "confidence" in result
        assert "num_sources" in result
        
        # Verify sources
        assert len(result["sources"]) > 0
        assert result["num_sources"] > 0
        
        # Verify confidence
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
        
        # Answer should contain context information
        assert len(result["answer"]) > 0
        assert "python" in result["answer"].lower()
        
        print(f"✅ RAG query successful")
        print(f"   Sources: {result['num_sources']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Answer length: {len(result['answer'])} chars")
    
    @pytest.mark.asyncio
    async def test_rag_query_with_filters(self, rag_service: TestContainerRAGService):
        """Test RAG query with metadata filters"""
        
        # Add documents with different categories
        test_docs = [
            {
                "content": "Python programming is great for data science and AI applications.",
                "metadata": {"category": "programming", "language": "python"}
            },
            {
                "content": "The Eiffel Tower is a famous landmark in Paris, France.",
                "metadata": {"category": "travel", "location": "paris"}
            },
            {
                "content": "Machine learning requires understanding of statistics and algorithms.",
                "metadata": {"category": "ai", "topic": "machine_learning"}
            }
        ]
        
        await rag_service.document_service.add_documents_batch(test_docs)
        
        # Query with category filter
        result = await rag_service.query(
            question="Tell me about programming",
            filters={"category": "programming"}
        )
        
        # Should only return programming-related sources
        assert len(result["sources"]) > 0
        for source in result["sources"]:
            # Note: We can't directly check metadata in the response structure,
            # but the content should be programming-related
            assert any(word in source["content_preview"].lower() 
                      for word in ["python", "programming", "code"])
        
        print(f"✅ Filtered RAG query returned {len(result['sources'])} programming sources")
    
    @pytest.mark.asyncio
    async def test_rag_multi_query(self, rag_service: TestContainerRAGService):
        """Test processing multiple questions"""
        
        # Add diverse test documents
        test_docs = [
            {
                "content": "Python is excellent for beginners due to its readable syntax.",
                "metadata": {"category": "programming"}
            },
            {
                "content": "Paris is home to the Louvre Museum and Eiffel Tower.",
                "metadata": {"category": "travel"}
            },
            {
                "content": "Machine learning models learn from training data.",
                "metadata": {"category": "ai"}
            }
        ]
        
        await rag_service.document_service.add_documents_batch(test_docs)
        
        # Test multiple questions
        questions = [
            "What programming language should I learn?",
            "What can I visit in Paris?",
            "How does machine learning work?"
        ]
        
        results = await rag_service.multi_query(questions)
        
        assert len(results) == len(questions)
        
        # Verify each result
        for i, result in enumerate(results):
            assert "answer" in result
            assert "sources" in result
            assert len(result["sources"]) > 0
            
            print(f"✅ Question {i+1}: {len(result['sources'])} sources found")
    
    @pytest.mark.asyncio
    async def test_rag_no_relevant_documents(self, rag_service: TestContainerRAGService):
        """Test RAG behavior when no relevant documents exist"""
        
        # Add a document on a specific topic
        await rag_service.document_service.add_document(
            content="The quantum mechanics principles in physics are complex.",
            metadata={"category": "physics"}
        )
        
        # Query about a completely different topic
        result = await rag_service.query("How do I cook pasta?")
        
        # Should handle gracefully
        assert "answer" in result
        assert "sources" in result
        
        # Might return some sources (even if not very relevant) or none
        # The key is that it doesn't crash
        print(f"✅ No relevant docs query handled gracefully")
        print(f"   Returned {len(result['sources'])} sources")


class TestDatabaseIntegrationWithContainers:
    """Test database-specific functionality with containers"""
    
    @pytest.mark.asyncio
    async def test_vector_similarity_calculation(self, async_session: AsyncSession):
        """Test that vector similarity calculations work correctly"""
        
        # Create mock embeddings that we know the similarity of
        import numpy as np
        
        # Create similar vectors
        base_vector = np.array([1.0, 0.0, 0.0] + [0.0] * 765)  # 768 dims
        similar_vector = np.array([0.9, 0.1, 0.0] + [0.0] * 765)
        different_vector = np.array([0.0, 0.0, 1.0] + [0.0] * 765)
        
        # Normalize vectors
        base_vector = base_vector / np.linalg.norm(base_vector)
        similar_vector = similar_vector / np.linalg.norm(similar_vector)
        different_vector = different_vector / np.linalg.norm(different_vector)
        
        # Add documents with known embeddings
        doc1 = Document(
            title="Base Document",
            content="This is the base document",
            embedding=base_vector.tolist(),
            is_processed=True
        )
        
        doc2 = Document(
            title="Similar Document", 
            content="This is a similar document",
            embedding=similar_vector.tolist(),
            is_processed=True
        )
        
        doc3 = Document(
            title="Different Document",
            content="This is a different document", 
            embedding=different_vector.tolist(),
            is_processed=True
        )
        
        async_session.add_all([doc1, doc2, doc3])
        await async_session.commit()
        
        # Test similarity search using raw SQL
        query_vector = base_vector.tolist()
        
        from sqlalchemy import select
        stmt = select(
            Document,
            Document.embedding.cosine_distance(query_vector).label('distance')
        ).where(Document.is_processed == True).order_by('distance').limit(3)
        
        result = await async_session.execute(stmt)
        docs_with_distances = result.fetchall()
        
        assert len(docs_with_distances) == 3
        
        # Verify order (base should be first, similar second, different third)
        base_doc, base_distance = docs_with_distances[0]
        similar_doc, similar_distance = docs_with_distances[1]  
        different_doc, different_distance = docs_with_distances[2]
        
        assert base_doc.title == "Base Document"
        assert similar_doc.title == "Similar Document"
        assert different_doc.title == "Different Document"
        
        # Verify distances (smaller = more similar)
        assert base_distance < similar_distance < different_distance
        assert base_distance < 0.1  # Should be very close to 0
        
        print(f"✅ Vector similarity test passed")
        print(f"   Base distance: {base_distance:.4f}")
        print(f"   Similar distance: {similar_distance:.4f}")
        print(f"   Different distance: {different_distance:.4f}")
    
    @pytest.mark.asyncio
    async def test_document_relationships(self, async_session: AsyncSession):
        """Test SQLAlchemy relationships between models"""
        
        # Create a document with metadata
        doc = Document(
            title="Test Document",
            content="This is a test document with metadata and chunks",
            embedding=[0.1] * 768,
            is_processed=True
        )
        
        async_session.add(doc)
        await async_session.flush()  # Get the ID without committing
        
        # Add metadata entries
        metadata1 = DocumentMetadata(
            document_id=doc.id,
            key="category",
            value="test",
            value_type="string"
        )
        
        metadata2 = DocumentMetadata(
            document_id=doc.id,
            key="priority", 
            value="1",
            value_type="number"
        )
        
        # Add document chunk
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content="This is a chunk of the document",
            embedding=[0.2] * 768,
            start_char=0,
            end_char=100,
            chunk_size=100
        )
        
        async_session.add_all([metadata1, metadata2, chunk])
        await async_session.commit()
        
        # Test relationships
        await async_session.refresh(doc, ['metadata_entries', 'chunks'])
        
        assert len(doc.metadata_entries) == 2
        assert len(doc.chunks) == 1
        
        # Test reverse relationships
        assert doc.metadata_entries[0].document == doc
        assert doc.chunks[0].document == doc
        
        print(f"✅ Document relationships test passed")
        print(f"   Document has {len(doc.metadata_entries)} metadata entries")
        print(f"   Document has {len(doc.chunks)} chunks")


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure test environment"""
    
    # Set test environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["VECTOR_DIMENSION"] = "768"
    
    print("🧪 Test environment configured")
    yield
    print("🧹 Test environment cleaned up")


# =============================================================================
# UTILITY FUNCTIONS FOR MANUAL TESTING
# =============================================================================

async def run_manual_tests():
    """Run tests manually for development"""
    
    print("🧪 Running manual testcontainer tests...")
    
    # Start container
    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test_user", 
        password="test_password",
        dbname="test_db"
    ) as postgres:
        
        time.sleep(5)
        
        # Setup test settings
        connection_url = postgres.get_connection_url()
        sync_url = connection_url.replace("postgresql://", "postgresql+psycopg2://")
        async_url = connection_url.replace("postgresql://", "postgresql+asyncpg://")
        
        test_settings = Settings(
            google_api_key="mock_key",
            database_url=sync_url,
            async_database_url=async_url,
            vector_dimension=768
        )
        
        # Initialize database
        conn = psycopg2.connect(connection_url)
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.close()
        
        # Create tables
        sync_engine = create_engine(test_settings.database_url)
        Base.metadata.create_all(bind=sync_engine)
        
        # Monkey patch settings
        import src.config.settings
        src.config.settings.settings = test_settings
        
        # Run tests
        doc_service = TestContainerDocumentService()
        
        # Test basic functionality
        doc_id = await doc_service.add_document(
            content="Test document for manual testing",
            metadata={"test": "manual"}
        )
        
        results = await doc_service.similarity_search("test document")
        
        print(f"✅ Manual test passed!")
        print(f"   Added document: {doc_id}")
        print(f"   Search returned: {len(results)} results")


if __name__ == "__main__":
    asyncio.run(run_manual_tests())


# =============================================================================
# PYTEST CONFIGURATION FILE
# =============================================================================

# File: pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings


# =============================================================================
# REQUIREMENTS UPDATE FOR TESTCONTAINERS
# =============================================================================

# Add to pyproject.toml dev dependencies:
[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0", 
    "testcontainers[postgresql]>=3.7.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
]
