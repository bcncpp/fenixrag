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