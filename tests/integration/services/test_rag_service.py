import pytest

from tests.integration.common import TestContainerRAGService


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
                "metadata": {
                    "category": "programming",
                    "difficulty": "beginner",
                },
            },
            {
                "title": "Machine Learning",
                "content": "Machine learning algorithms learn patterns from data to make predictions.",
                "metadata": {"category": "ai", "difficulty": "intermediate"},
            },
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

        print("✅ RAG query successful")
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
                "metadata": {"category": "programming", "language": "python"},
            },
            {
                "content": "The Eiffel Tower is a famous landmark in Paris, France.",
                "metadata": {"category": "travel", "location": "paris"},
            },
            {
                "content": "Machine learning requires understanding of statistics and algorithms.",
                "metadata": {"category": "ai", "topic": "machine_learning"},
            },
        ]

        await rag_service.document_service.add_documents_batch(test_docs)

        # Query with category filter
        result = await rag_service.query(
            question="Tell me about programming",
            filters={"category": "programming"},
        )

        # Should only return programming-related sources
        assert len(result["sources"]) > 0
        for source in result["sources"]:
            # Note: We can't directly check metadata in the response structure,
            # but the content should be programming-related
            assert any(
                word in source["content_preview"].lower() for word in ["python", "programming", "code"]
            )

        print(f"✅ Filtered RAG query returned {len(result['sources'])} programming sources")

    @pytest.mark.asyncio
    async def test_rag_multi_query(self, rag_service: TestContainerRAGService):
        """Test processing multiple questions"""
        # Add diverse test documents
        test_docs = [
            {
                "content": "Python is excellent for beginners due to its readable syntax.",
                "metadata": {"category": "programming"},
            },
            {
                "content": "Paris is home to the Louvre Museum and Eiffel Tower.",
                "metadata": {"category": "travel"},
            },
            {
                "content": "Machine learning models learn from training data.",
                "metadata": {"category": "ai"},
            },
        ]

        await rag_service.document_service.add_documents_batch(test_docs)

        # Test multiple questions
        questions = [
            "What programming language should I learn?",
            "What can I visit in Paris?",
            "How does machine learning work?",
        ]

        results = await rag_service.multi_query(questions)

        assert len(results) == len(questions)

        # Verify each result
        for i, result in enumerate(results):
            assert "answer" in result
            assert "sources" in result
            assert len(result["sources"]) > 0

            print(f"✅ Question {i + 1}: {len(result['sources'])} sources found")

    @pytest.mark.asyncio
    async def test_rag_no_relevant_documents(self, rag_service: TestContainerRAGService):
        """Test RAG behavior when no relevant documents exist"""
        # Add a document on a specific topic
        await rag_service.document_service.add_document(
            content="The quantum mechanics principles in physics are complex.",
            metadata={"category": "physics"},
        )

        # Query about a completely different topic
        result = await rag_service.query("How do I cook pasta?")

        # Should handle gracefully
        assert "answer" in result
        assert "sources" in result

        # Might return some sources (even if not very relevant) or none
        # The key is that it doesn't crash
        print("✅ No relevant docs query handled gracefully")
        print(f"   Returned {len(result['sources'])} sources")
